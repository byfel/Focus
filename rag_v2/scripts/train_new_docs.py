#!/usr/bin/env python3
"""
ETAPA 5 DO PIPELINE: Treinamento Incremental para Novos Documentos (Fine-Tuning).

Integra-se ao fluxo completo do FOCUS:
    1. ingest.py            -> Extrai texto e vision dos PDFs para chunks.json
    2. embed.py             -> Gera embeddings incrementais (V1)
    3. migrate_to_qdrant.py -> Sincroniza Qdrant (V1)
    4. populate_v2.py       -> Gera embeddings nomic-embed-text e sincroniza Qdrant (V2)
    5. train_new_docs.py    -> Sintetiza casos de treino dos novos documentos,
                               atualiza o Fine-Tuning LoRA e publica no Ollama!

Uso:
    # Execução padrão (detecta novos documentos, gera dataset, treina e publica no Ollama):
    python3 scripts/train_new_docs.py

    # Apenas atualizar o dataset com os novos documentos (sem rodar GPU train agora):
    python3 scripts/train_new_docs.py --dataset-only

    # Treinar usando as 8 GPUs Quadro M6000 via Accelerate:
    python3 scripts/train_new_docs.py --num-gpus 8
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path

PROJECT_V2_ROOT = Path(__file__).resolve().parent.parent
FINE_TUNING_DIR = PROJECT_V2_ROOT / "fine_tuning"
DATASET_FILE = FINE_TUNING_DIR / "dataset" / "media_solver_train.jsonl"
DATASET_STATE_FILE = FINE_TUNING_DIR / "dataset" / "dataset_state.json"
TRAIN_STATE_FILE = FINE_TUNING_DIR / "output" / "train_state.json"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_step(cmd: list, description: str) -> int:
    print(f"\n▶️  {description}")
    print(f"   Comando: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ Erro na etapa '{description}' (código de saída: {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Etapa 5: Aprendizado e Fine-Tuning Incremental para Novos Documentos"
    )
    parser.add_argument("--chunks-per-doc", type=int, default=8,
                        help="Quantos chunks representativos aprender de cada documento (0 = todos). Padrão: 8")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limite máximo de chunks a sintetizar nesta execução (0 = todos os pendentes)")
    parser.add_argument("--only-new-docs", action="store_true",
                        help="Sintetizar apenas documentos 100%% novos (ignorar docs parcialmente processados)")
    parser.add_argument("--mark-learned", type=str, nargs="*", default=None,
                        help="Marcar documentos específicos como já aprendidos no estado")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Apenas gerar/atualizar o dataset incremental, sem iniciar o treino LoRA")
    parser.add_argument("--skip-export", action="store_true",
                        help="Não exportar automaticamente para o Ollama após o treino")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Número de GPUs para o treinamento LoRA (ex: 1 ou 8 para usar accelerate)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Número de épocas de treino LoRA (padrão: 3)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Modelo base HuggingFace para o LoRA (padrão: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--ollama-model-name", type=str, default="focus-media-solver",
                        help="Nome do modelo final no Ollama")
    parser.add_argument("--force-train", action="store_true",
                        help="Forçar execução do treinamento mesmo sem novos exemplos gerados")
    parser.add_argument("--status", action="store_true",
                        help="Apenas exibe o relatório de cobertura de todos os documentos e status do último treino")
    args = parser.parse_args()

    print("=" * 80)
    print("🧠 ETAPA 5: APRENDIZADO E TREINAMENTO INCREMENTAL DE DOCUMENTOS (FOCUS AI)")
    print("=" * 80)

    before_count = count_jsonl_lines(DATASET_FILE)
    before_hash = file_sha256(DATASET_FILE)

    # -------------------------------------------------------------------------
    # PASSO 5.1: Síntese Incremental de Dataset (Todos os docs / Novos docs)
    # -------------------------------------------------------------------------
    gen_script = FINE_TUNING_DIR / "generate_dataset.py"
    gen_cmd = [
        sys.executable, str(gen_script),
        "--source", "all",
        "--chunks-per-doc", str(args.chunks_per_doc),
    ]
    if args.status:
        gen_cmd.append("--status")
        sys.exit(subprocess.run(gen_cmd).returncode)
    if args.max_samples > 0:
        gen_cmd.extend(["--max-samples", str(args.max_samples)])
    if args.only_new_docs:
        gen_cmd.append("--only-new-docs")
    if args.mark_learned:
        gen_cmd.append("--mark-learned")
        gen_cmd.extend(args.mark_learned)

    rc = run_step(gen_cmd, "Passo 5.1: Gerando casos de estudo para documentos novos/pendentes")
    if rc != 0:
        sys.exit(rc)

    after_count = count_jsonl_lines(DATASET_FILE)
    after_hash = file_sha256(DATASET_FILE)
    added_examples = max(0, after_count - before_count)

    print("\n" + "-" * 80)
    print(f"📊 Resumo do Dataset ({DATASET_FILE.name}):")
    print(f"   Exemplos antes  : {before_count}")
    print(f"   Exemplos agora  : {after_count} (+{added_examples} novos)")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # PASSO 5.1b: Sincroniza casos de TSHOOT e resoluções no Qdrant V2 (Híbrido)
    # -------------------------------------------------------------------------
    populate_v2_script = PROJECT_V2_ROOT / "scripts" / "populate_v2.py"
    if populate_v2_script.exists():
        run_step(
            [sys.executable, str(populate_v2_script)],
            "Passo 5.1b: Indexando resoluções e casos TSHOOT_KB no Qdrant V2 (RAG Híbrido)"
        )

    after_hash = file_sha256(DATASET_FILE)

    if args.dataset_only:
        print("\n✅ Modo --dataset-only concluído! O dataset e o Qdrant V2 estão atualizados.")
        return

    # Verifica se o dataset mudou em relação ao último treinamento LoRA realizado
    last_trained_hash = ""
    if TRAIN_STATE_FILE.exists():
        try:
            last_trained_hash = json.loads(TRAIN_STATE_FILE.read_text(encoding="utf-8")).get("dataset_sha256", "")
        except Exception:
            pass

    needs_training = (after_hash != last_trained_hash) or args.force_train

    if not needs_training:
        print("\n✨ Nenhum novo documento ou exemplo pendente de treinamento LoRA.")
        print("   O modelo fine-tuned já contempla todos os exemplos atuais do dataset!")
        return

    # -------------------------------------------------------------------------
    # PASSO 5.2: Treinamento Supervisionado LoRA (SFT)
    # -------------------------------------------------------------------------
    train_script = FINE_TUNING_DIR / "train.py"
    if args.num_gpus > 1:
        train_cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", str(args.num_gpus),
            str(train_script),
            "--base-model", args.base_model,
            "--epochs", str(args.epochs),
        ]
    else:
        train_cmd = [
            sys.executable, str(train_script),
            "--base-model", args.base_model,
            "--epochs", str(args.epochs),
        ]

    if args.force_train:
        train_cmd.append("--force")

    rc = run_step(train_cmd, f"Passo 5.2: Executando Fine-Tuning LoRA ({args.num_gpus} GPU(s))")
    if rc != 0:
        sys.exit(rc)

    # -------------------------------------------------------------------------
    # PASSO 5.3: Merge LoRA e Deploy no Ollama
    # -------------------------------------------------------------------------
    if not args.skip_export:
        export_script = FINE_TUNING_DIR / "export_to_ollama.py"
        export_cmd = [
            sys.executable, str(export_script),
            "--base-model", args.base_model,
            "--model-name", args.ollama_model_name,
        ]
        rc = run_step(export_cmd, f"Passo 5.3: Atualizando modelo '{args.ollama_model_name}' no Ollama")
        if rc != 0:
            sys.exit(rc)

    print("\n" + "=" * 80)
    print("🎉 ETAPA 5 CONCLUÍDA! O modelo aprendeu os novos documentos com sucesso.")
    print("=" * 80)


if __name__ == "__main__":
    main()
