#!/usr/bin/env python3
"""
FOCUS AI - Orquestrador Completo de Ingestão, Indexação e Treinamento Incremental (5 Etapas).

Executa em sequência todo o pipeline para novos documentos:
  [Etapa 1/5] ingest.py            -> Extração de texto e Vision dos PDFs para chunks.json
  [Etapa 2/5] embed.py             -> Geração incremental de embeddings V1 (embeddings.json)
  [Etapa 3/5] migrate_to_qdrant.py -> Sincronização incremental no Qdrant V1 (rag_documents)
  [Etapa 4/5] populate_v2.py       -> Geração incremental de embeddings Nomic e Qdrant V2 (rag_documents_v2)
  [Etapa 5/5] train_new_docs.py    -> Geração incremental de Dataset + Fine-Tuning LoRA + Deploy Ollama

Uso:
    # Pipeline completo (5 etapas):
    python3 scripts/run_full_pipeline.py --scope all --num-gpus 8

    # Pipeline completo gerando dataset dos novos documentos, mas pulando o treino GPU agora:
    python3 scripts/run_full_pipeline.py --dataset-only
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_V2_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_V2_ROOT.parent


def resolve_v1_script(script_name: str) -> Path:
    """Localiza o script da V1 tanto no servidor (/opt/ai/rag/scripts) quanto no repositório local."""
    candidates = [
        Path("/opt/ai/rag/scripts") / script_name,
        REPO_ROOT / "rag" / "scripts" / script_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def run_stage(stage_num: int, total_stages: int, title: str, cmd: list):
    print("\n" + "=" * 80)
    print(f"🚀 [ETAPA {stage_num}/{total_stages}] {title}")
    print(f"   Comando: {' '.join(str(c) for c in cmd)}")
    print("=" * 80 + "\n")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n❌ Falha na Etapa {stage_num} ({title}). Código de retorno: {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n✅ [ETAPA {stage_num}/{total_stages}] Concluída em {elapsed:.1f}s!")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador Completo FOCUS AI: ingest -> embed -> migrate_qdrant -> populate_v2 -> train_new_docs"
    )
    parser.add_argument("--scope", choices=["public", "internal", "all"], default="all",
                        help="Escopo para o ingest.py (padrão: all)")
    parser.add_argument("--skip-v1-qdrant", action="store_true",
                        help="Pular embed.py e migrate_to_qdrant.py da V1 (executar apenas ingest + populate_v2 + train)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Pular a Etapa 5 de treinamento")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Na Etapa 5, apenas atualizar o dataset com os novos documentos sem iniciar o treino LoRA")
    parser.add_argument("--chunks-per-doc", type=int, default=8,
                        help="Quantidade de chunks representativos por documento para o Fine-Tuning (padrão: 8, 0 = todos)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Quantidade de GPUs para o Fine-Tuning LoRA (ex: 1 ou 8)")
    args = parser.parse_args()

    ingest_script = resolve_v1_script("ingest.py")
    embed_script = resolve_v1_script("embed.py")
    migrate_script = resolve_v1_script("migrate_to_qdrant.py")
    populate_v2_script = PROJECT_V2_ROOT / "scripts" / "populate_v2.py"
    train_new_docs_script = PROJECT_V2_ROOT / "scripts" / "train_new_docs.py"

    total_stages = 5
    start_all = time.time()

    # 1. Ingest
    run_stage(
        1, total_stages,
        "INGESTÃO DE DOCUMENTOS (PDF Text + Vision -> chunks.json)",
        [sys.executable, str(ingest_script), "--scope", args.scope]
    )

    # 2 & 3. Embed V1 + Migrate to Qdrant V1
    if not args.skip_v1_qdrant:
        run_stage(
            2, total_stages,
            "GERAÇÃO INCREMENTAL DE EMBEDDINGS V1 (embeddings.json)",
            [sys.executable, str(embed_script)]
        )
        run_stage(
            3, total_stages,
            "MIGRAÇÃO INCREMENTAL PARA QDRANT V1 (rag_documents)",
            [sys.executable, str(migrate_script)]
        )
    else:
        print("\n⏭️  Pulando Etapas 2 e 3 (V1) conforme --skip-v1-qdrant.")

    # 4. Populate V2
    run_stage(
        4, total_stages,
        "POPULAÇÃO INCREMENTAL QDRANT V2 (nomic-embed-text -> rag_documents_v2)",
        [sys.executable, str(populate_v2_script)]
    )

    # 5. Train New Docs (Dataset Incremental + LoRA + Ollama)
    if not args.skip_train:
        train_cmd = [
            sys.executable, str(train_new_docs_script),
            "--chunks-per-doc", str(args.chunks_per_doc),
            "--num-gpus", str(args.num_gpus),
        ]
        if args.dataset_only:
            train_cmd.append("--dataset-only")

        run_stage(
            5, total_stages,
            "TREINAMENTO INCREMENTAL PARA NOVOS DOCUMENTOS (Fine-Tuning + Ollama)",
            train_cmd
        )
    else:
        print("\n⏭️  Pulando Etapa 5 (--skip-train).")

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE COMPLETO FINALIZADO COM SUCESSO EM {total_elapsed / 60:.1f} MINUTOS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
