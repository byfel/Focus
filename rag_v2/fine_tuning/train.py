#!/usr/bin/env python3
"""
FOCUS AI - Treinamento Incremental de Fine-Tuning (LoRA / SFT) para Solucionador de Mídia.

Otimizado para servidor multi-GPU com 8x NVIDIA Quadro M6000 12GB (Maxwell sm_52):
- Precisão: FP16 (sem BF16, não suportado por Maxwell)
- Mecanismo de Atenção: Eager / SDPA nativo (sem FlashAttention-2)
- Multi-GPU: Suporta Hugging Face Accelerate, DDP ou execução em 1 GPU
- Aprendizado Contínuo (Incremental): Se já existir um adaptador LoRA treinado em
  `output/lora_adapter`, retoma/refina os pesos existentes com os novos documentos,
  e registra o hash do dataset em `output/train_state.json` para não retreinar
  desnecessariamente se não houver documentos novos.

Como executar no servidor (Exemplos):
    # Execução em 1 GPU:
    python3 train.py --epochs 3 --batch-size 1

    # Execução em todas as 8 GPUs (via accelerate):
    accelerate launch --multi_gpu --num_processes 8 train.py --epochs 3 --batch-size 1
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path

# Em servidores multi-GPU (ex: NVIDIA Quadro VCA 6000), a comunicação P2P PCIe entre
# sockets diferentes trava em broadcast_coalesced ("the launch timed out").
# Isolamos o treinamento na GPU 1 dedicada (12 GB VRAM) para execução limpa em 1 GPU.
if "CUDA_VISIBLE_DEVICES" not in os.environ and Path("/dev/nvidia1").exists():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("focus_train")


def calculate_file_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def sanitize_jsonl_dataset(file_path: Path) -> int:
    """
    Garante que todas as linhas do .jsonl tenham 'messages' válidos e que o campo
    'content' de cada mensagem seja sempre uma string (corrige objetos/listas aninhados).
    Retorna a quantidade de linhas corrigidas.
    """
    if not file_path.exists():
        return 0

    fixed_count = 0
    valid_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    fixed_count += 1
                    continue
                line_modified = False
                clean_msgs = []
                for m in msgs:
                    if not isinstance(m, dict):
                        line_modified = True
                        continue
                    role = str(m.get("role", "user")).strip()
                    content = m.get("content", "")
                    if not isinstance(content, str):
                        line_modified = True
                        if isinstance(content, dict):
                            parts = []
                            for k, v in content.items():
                                if isinstance(v, list):
                                    v_str = "\n".join(f"- {item}" for item in v)
                                else:
                                    v_str = str(v)
                                parts.append(f"{k}:\n{v_str}")
                            content = "\n\n".join(parts)
                        elif isinstance(content, list):
                            content = "\n".join(str(x) for x in content)
                        else:
                            content = str(content)
                    clean_msgs.append({"role": role, "content": content.strip()})
                if line_modified:
                    fixed_count += 1
                valid_lines.append(json.dumps({"messages": clean_msgs}, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"Linha {line_num} inválida ignorada no dataset: {e}")
                fixed_count += 1

    if fixed_count > 0:
        logger.info(f"🧹 Sanitizando dataset: {fixed_count} registro(s) com 'content' não-string corrigidos em {file_path.name}")
        with open(file_path, "w", encoding="utf-8") as f:
            for vl in valid_lines:
                f.write(vl + "\n")

    return fixed_count


def check_gpu_environment():
    """Valida se GPUs estão disponíveis e exibe detalhes de memória."""
    if not torch.cuda.is_available():
        logger.warning("⚠️  Nenhuma GPU NVIDIA detectada! O treinamento em CPU será extremamente lento.")
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"🚀 GPUs Detectadas: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability(i)
        logger.info(f"   [GPU {i}] {gpu_name} | VRAM: {vram_gb:.1f} GB | Compute Capability: {cap[0]}.{cap[1]}")


def main():
    parser = argparse.ArgumentParser(description="FOCUS AI - Fine-Tuning Incremental de Solucionador de Mídia")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Modelo base Hugging Face (padrão: Qwen/Qwen2.5-1.5B-Instruct, otimizado para VCA 6000)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Caminho do arquivo .jsonl gerado")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Diretório onde salvar os pesos LoRA treinados")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Número de épocas (padrão: 3)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size por GPU (padrão: 1 para 12GB de VRAM)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Passos de acumulação de gradiente (padrão: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (padrão: 1e-4)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Tamanho máximo de contexto de treino (tokens, padrão: 512)")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="Rank da matriz LoRA (padrão: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="Alpha da matriz LoRA (padrão: 16)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Treinar um novo adaptador LoRA do zero em vez de continuar do adaptador existente")
    parser.add_argument("--force", action="store_true",
                        help="Forçar treinamento mesmo que o dataset não tenha mudado desde o último treino")
    args = parser.parse_args()

    default_data = Path(__file__).resolve().parent / "dataset" / "media_solver_train.jsonl"
    dataset_path = Path(args.dataset) if args.dataset else default_data
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "output" / "lora_adapter"
    train_state_file = output_dir.parent / "train_state.json"

    if not dataset_path.exists():
        logger.error(f"Arquivo de dataset não encontrado: {dataset_path}")
        logger.error("Execute primeiro: python3 generate_dataset.py --source chunks")
        sys.exit(1)

    sanitize_jsonl_dataset(dataset_path)
    dataset_hash = calculate_file_sha256(dataset_path)
    adapter_config_file = output_dir / "adapter_config.json"

    # Verifica se já treinou exatamente este dataset
    if not args.force and adapter_config_file.exists() and train_state_file.exists():
        try:
            prev_state = json.loads(train_state_file.read_text(encoding="utf-8"))
            if prev_state.get("dataset_sha256") == dataset_hash:
                logger.info("✨ O dataset não teve novos exemplos desde o último treinamento.")
                logger.info(f"   Último treino: {prev_state.get('timestamp')} ({prev_state.get('samples_count')} exemplos)")
                logger.info("   Use --force caso queira retreinar mesmo assim.")
                return
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)

    check_gpu_environment()

    # Detecta se a GPU é arquitetura Maxwell (sm_52, Compute Capability < 6.0):
    # Maxwell não possui unidades FP16 nativas (FP16 é emulado em software, causando
    # lentidão extrema, 'launch timed out' no watchdog e 'grad_norm: nan').
    # Já em FP32 nativo (cublasSgemm), cada VCA 6000 entrega 7 TFLOPS reais sem overflow,
    # e as 7 GPUs dedicadas somam 84 GB de VRAM (o modelo 7B ocupa ~4.3 GB por GPU).
    is_maxwell = False
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        if cap_major < 6:
            is_maxwell = True

    model_dtype = torch.float32 if is_maxwell else torch.float16
    use_fp16_amp = False if is_maxwell else True

    logger.info(f"Carregando dataset de: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    logger.info(f"Total de amostras de treino: {len(dataset)}")

    logger.info(f"Carregando Tokenizer do modelo base: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_label = "FP32 Nativo (Alta Velocidade p/ Maxwell sm_52)" if is_maxwell else "FP16"
    logger.info(f"Carregando Modelo Base ({args.base_model}) em {dtype_label}...")
    model_kwargs = {
        "torch_dtype": model_dtype,
        "attn_implementation": "eager",
        "trust_remote_code": True,
    }
    # Para modelos compactos (1.5B / 0.5B), aloca inteiro em 1 única GPU dedicada (sem hooks multi-GPU)
    # evitando gargalos de P2P PCIe e bugs de gradiente NaN entre GPUs.
    if os.environ.get("LOCAL_RANK") is None and torch.cuda.is_available():
        if any(sz in args.base_model.upper() for sz in ["0.5B", "1.5B", "1.8B"]):
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    if adapter_config_file.exists() and not args.from_scratch:
        logger.info(f"♻️  Adaptador LoRA anterior encontrado em {output_dir}. Continuando aprendizado incremental...")
        model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
    else:
        logger.info("Aplicando novo adaptador LoRA ao modelo...")
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Argumentos de Treino (gradient_checkpointing=False para evitar 'launch timed out' no watchdog da VCA)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=use_fp16_amp,
        bf16=False,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
    )

    logger.info("🚀 Iniciando treinamento supervisionado (SFT)...")
    start_t = time.time()
    trainer.train()
    elapsed = time.time() - start_t

    logger.info(f"💾 Salvando pesos finais do adaptador LoRA em: {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Registra estado do treinamento concluído
    train_state = {
        "base_model": args.base_model,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "samples_count": len(dataset),
        "epochs": args.epochs,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    train_state_file.write_text(json.dumps(train_state, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"✅ Treinamento concluído com sucesso em {elapsed / 60:.1f} min!")
    logger.info("Próximo passo: execute 'export_to_ollama.py' para atualizar o modelo no Ollama.")


if __name__ == "__main__":
    main()
