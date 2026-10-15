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
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Modelo base Hugging Face (padrão: Qwen/Qwen2.5-7B-Instruct)")
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
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (padrão: 2e-4)")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Tamanho máximo de contexto de treino (tokens)")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="Rank da matriz LoRA (padrão: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="Alpha da matriz LoRA (padrão: 32)")
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

    logger.info(f"Carregando dataset de: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    logger.info(f"Total de amostras de treino: {len(dataset)}")

    logger.info(f"Carregando Tokenizer do modelo base: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Carregando Modelo Base em FP16 (Otimizado para Quadro M6000 / Maxwell)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    if adapter_config_file.exists() and not args.from_scratch:
        logger.info(f"♻️  Adaptador LoRA anterior encontrado em {output_dir}. Continuando aprendizado incremental...")
        model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
        peft_for_trainer = None
    else:
        logger.info("Aplicando novo adaptador LoRA ao modelo...")
        model = get_peft_model(model, peft_config)
        peft_for_trainer = peft_config

    model.print_trainable_parameters()

    # Argumentos de Treino (compatíveis com Maxwell)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,                       # FP16 ativado para as M6000
        bf16=False,                      # BF16 desativado (não suportado em Maxwell)
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",                # Sem dependência de wandb/comet
        gradient_checkpointing=True,     # Economiza VRAM nas M6000 de 12GB
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    if peft_for_trainer is not None:
        trainer_kwargs["peft_config"] = peft_for_trainer

    trainer = SFTTrainer(**trainer_kwargs)

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
