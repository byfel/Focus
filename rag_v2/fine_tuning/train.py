#!/usr/bin/env python3
"""
FOCUS AI - Treinamento de Fine-Tuning (LoRA / SFT) para Solucionador de Mídia.

Otimizado para servidor multi-GPU com 8x NVIDIA Quadro M6000 12GB (Maxwell sm_52):
- Precisão: FP16 (sem BF16, não suportado por Maxwell)
- Mecanismo de Atenção: Eager / SDPA nativo (sem FlashAttention-2)
- Multi-GPU: Suporta Hugging Face Accelerate, DDP ou device_map='auto'
- Formato: ChatML / Messages (SFTTrainer do TRL)

Como executar no servidor (Exemplos):
    # Execução em 1 GPU (para teste rápido):
    python3 train.py --epochs 3 --batch-size 1

    # Execução em todas as 8 GPUs (via torchrun / accelerate):
    accelerate launch --multi_gpu --num_processes 8 train.py --epochs 5 --batch-size 1
"""

import os
import sys
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
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("focus_train")


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
    parser = argparse.ArgumentParser(description="FOCUS AI - Fine-Tuning de Solucionador de Mídia")
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
    args = parser.parse_args()

    default_data = Path(__file__).resolve().parent / "dataset" / "media_solver_train.jsonl"
    dataset_path = Path(args.dataset) if args.dataset else default_data
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "output" / "lora_adapter"

    if not dataset_path.exists():
        logger.error(f"Arquivo de dataset não encontrado: {dataset_path}")
        logger.error("Execute primeiro: python3 generate_dataset.py --source files")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    check_gpu_environment()

    logger.info(f"Carregando dataset de: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    logger.info(f"Total de amostras de treino: {len(dataset)}")

    logger.info(f"Carregando Tokenizer do modelo base: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Carregando Modelo Base em FP16 (Otimizado para Quadro M6000 / Maxwell)...")
    # Para Maxwell sm_52:
    # 1. torch_dtype=torch.float16 (sem bf16)
    # 2. attn_implementation="eager" (evita FlashAttention-2 que falharia em sm_52)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # Configuração LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    logger.info("Aplicando adaptador LoRA ao modelo...")
    model = get_peft_model(model, peft_config)
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    logger.info("🚀 Iniciando treinamento supervisionado (SFT)...")
    trainer.train()

    logger.info(f"💾 Salvando pesos finais do adaptador LoRA em: {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("✅ Treinamento concluído com sucesso!")
    logger.info("Próximo passo: execute 'export_to_ollama.py' para gerar o GGUF e carregar no Ollama.")


if __name__ == "__main__":
    main()
