#!/usr/bin/env python3
"""
Popula a coleção rag_documents_v2 com embeddings gerados pelo nomic-embed-text.

Integra automaticamente 3 fontes de conhecimento no Qdrant V2:
  1. chunks.json (textos e descrições Vision extraídos dos PDFs pela V1)
  2. Documentos e Resoluções de TSHOOT locais (.md, .txt, .pdf em data/documents e /opt/ai/rag/documents)
  3. Base de Casos Estruturados de Troubleshooting (media_solver_train.jsonl), já sanitizada
     de dicionário para Markdown limpo!

Uso:
    python3 scripts/populate_v2.py
    python3 scripts/populate_v2.py --recreate   # recria a coleção do zero
"""

import sys
import os
import ast
import json
import time
import hashlib
import argparse
from pathlib import Path

# Adiciona o path da V2
PROJECT_V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_V2_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from core.config import (
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE,
    DATA_ROOT, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
)
from core.embeddings import get_embeddings_batch
from core.generator import format_solver_output_to_markdown

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CHUNKS_FILE = os.path.join(DATA_ROOT, "database", "chunks.json")
DATASET_JSONL_FILE = PROJECT_V2_ROOT / "fine_tuning" / "dataset" / "media_solver_train.jsonl"


def generate_qdrant_id(chunk_id: str) -> int:
    """Gera ID determinístico (int positivo de 63 bits) para o Qdrant."""
    value = int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16], 16)
    return value & 0x7FFFFFFFFFFFFFFF


def create_collection(client: QdrantClient, recreate: bool = False):
    """Cria a coleção rag_documents_v2 no Qdrant."""
    collections = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION in collections:
        if recreate:
            print(f"🗑️  Removendo coleção existente: {QDRANT_COLLECTION}")
            client.delete_collection(QDRANT_COLLECTION)
        else:
            info = client.get_collection(QDRANT_COLLECTION)
            print(f"✅ Coleção '{QDRANT_COLLECTION}' já existe ({info.points_count} pontos)")
            return

    print(f"📦 Criando coleção: {QDRANT_COLLECTION}")
    print(f"   Dimensão: {QDRANT_VECTOR_SIZE}, Distância: Cosine")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )
    print(f"   ✅ Coleção criada com sucesso!")


def load_local_resolution_files(known_docs: set) -> list:
    """
    Escaneia diretórios de documentos por arquivos .md e .txt (ex: guias de TSHOOT e resoluções)
    para que entrem automaticamente no Qdrant V2 sem precisar serem PDFs.
    """
    extra_chunks = []
    candidate_dirs = [
        PROJECT_V2_ROOT / "data" / "documents",
        Path(DATA_ROOT) / "documents",
        Path("/opt/ai/rag/documents"),
    ]
    visited_dirs = set()

    for d in candidate_dirs:
        if not d.exists():
            continue
        real_d = str(d.resolve())
        if real_d in visited_dirs:
            continue
        visited_dirs.add(real_d)

        for fp in sorted(d.glob("**/*")):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext not in (".md", ".txt"):
                continue
            if fp.name in known_docs:
                continue

            try:
                content = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue

            if len(content) < 80:
                continue

            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            blocks = [content[i:i + 1800] for i in range(0, len(content), 1800)]
            for b_idx, block in enumerate(blocks, start=1):
                if len(block.strip()) < 50:
                    continue
                cid = hashlib.sha256(f"local_doc|{fp.name}|{b_idx}|{block}".encode("utf-8")).hexdigest()
                extra_chunks.append({
                    "chunk_id": cid,
                    "document": fp.name,
                    "file_hash": file_hash,
                    "page": b_idx,
                    "chunk": b_idx,
                    "text": block.strip(),
                })
    return extra_chunks


def sanitize_and_load_tshoot_dataset() -> list:
    """
    Lê media_solver_train.jsonl, converte respostas que estavam como dict Python ('SINTOMA': ...)
    para Markdown técnico limpo (atualizando o próprio .jsonl se necessário) e retorna cada caso
    de estudo como um chunk estruturado para indexação direta no Qdrant V2.
    """
    if not DATASET_JSONL_FILE.exists():
        return []

    tshoot_chunks = []
    updated_lines = []
    cleaned_count = 0

    try:
        raw_lines = DATASET_JSONL_FILE.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"⚠️ Não foi possível ler {DATASET_JSONL_FILE}: {e}")
        return []

    for idx, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            user_q = ""
            assistant_a = ""
            new_msgs = []
            for m in msgs:
                role = m.get("role", "")
                content = str(m.get("content", "")).strip()
                if role == "user":
                    user_q = content
                elif role == "assistant":
                    if content.startswith("{"):
                        clean_md = format_solver_output_to_markdown(content)
                        if clean_md != content:
                            content = clean_md
                            cleaned_count += 1
                    assistant_a = content
                new_msgs.append({"role": role, "content": content})

            updated_lines.append(json.dumps({"messages": new_msgs}, ensure_ascii=False))

            if user_q and assistant_a:
                combined_text = (
                    f"CASO DE TROUBLESHOOTING / RESOLUÇÃO TÉCNICA:\n"
                    f"PERGUNTA / INCIDENTE: {user_q}\n\n"
                    f"{assistant_a}"
                )
                cid = hashlib.sha256(f"tshoot_kb|{user_q}".encode("utf-8")).hexdigest()
                tshoot_chunks.append({
                    "chunk_id": cid,
                    "document": "TSHOOT_KB (Base de Resoluções)",
                    "file_hash": cid[:16],
                    "page": idx,
                    "chunk": 1,
                    "text": combined_text,
                })
        except Exception:
            updated_lines.append(line)

    if cleaned_count > 0:
        try:
            DATASET_JSONL_FILE.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
            print(f"   🧹 {cleaned_count} casos em {DATASET_JSONL_FILE.name} convertidos de dict '{{...}}' para Markdown limpo!")
        except Exception:
            pass

    return tshoot_chunks


def remove_synthetic_tshoot_kb(client: QdrantClient):
    """Remove pontos sintéticos antigos 'TSHOOT_KB (Base de Resoluções)' que diluíam os manuais reais."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document",
                        match=MatchValue(value="TSHOOT_KB (Base de Resoluções)"),
                    )
                ]
            ),
            wait=True,
        )
    except Exception:
        pass


def load_chunks() -> list:
    """Carrega chunks.json (PDFs reais) + arquivos .md/.txt reais de TSHOOT."""
    chunks = []
    known_docs = set()

    print(f"\n📂 [1/2] Carregando chunks de manuais oficiais: {CHUNKS_FILE}")
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            pdf_chunks = json.load(f)
        chunks.extend(pdf_chunks)
        for c in pdf_chunks:
            if c.get("document"):
                known_docs.add(c["document"])
        print(f"   ✅ {len(pdf_chunks)} chunks carregados de chunks.json")
    else:
        print(f"   ⚠️ Arquivo chunks.json não encontrado em {CHUNKS_FILE}")

    print(f"📂 [2/2] Verificando guias e resoluções reais (.md / .txt)...")
    local_md_chunks = load_local_resolution_files(known_docs)
    if local_md_chunks:
        chunks.extend(local_md_chunks)
        print(f"   ✅ +{len(local_md_chunks)} chunks de guias .md/.txt adicionados")
    else:
        print("   ℹ️ Nenhum arquivo .md/.txt adicional fora do chunks.json")

    # Higieniza o .jsonl para treinos futuros sem injetar resumos sintéticos no Qdrant
    sanitize_and_load_tshoot_dataset()

    # Mostra resumo por documento
    docs = {}
    for c in chunks:
        doc = c.get("document", "desconhecido")
        docs[doc] = docs.get(doc, 0) + 1

    print(f"\n   📄 Total consolidado: {len(chunks)} chunks em {len(docs)} documentos oficiais:")
    for doc, count in sorted(docs.items()):
        print(f"      • {doc}: {count} chunks")

    return chunks


def get_existing_ids(client: QdrantClient) -> set:
    """Recupera IDs de pontos já existentes no Qdrant para ingestão incremental."""
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        return set()

    ids = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for record in records:
            ids.add(record.id)
        if offset is None:
            break
    return ids


def populate(client: QdrantClient, chunks: list, incremental: bool = True):
    """Gera embeddings com nomic-embed-text e insere no Qdrant."""
    if incremental:
        existing_ids = get_existing_ids(client)
        if existing_ids:
            print(f"🔍 Verificando pontos existentes: {len(existing_ids)} encontrados.")
            chunks_to_process = [
                c for c in chunks
                if generate_qdrant_id(c.get("chunk_id", "")) not in existing_ids
            ]
            if not chunks_to_process:
                print("✨ Todos os chunks (manuais + TSHOOT_KB) já estão indexados no Qdrant v2!")
                return 0, 0, 0
            print(f"🆕 {len(chunks_to_process)} chunks novos (manuais/resoluções) a serem indexados.")
            chunks = chunks_to_process

    total = len(chunks)
    batch_size = EMBEDDING_BATCH_SIZE
    total_batches = (total + batch_size - 1) // batch_size

    print(f"\n🚀 Gerando embeddings com {EMBEDDING_MODEL}")
    print(f"   Lotes: {total_batches} x {batch_size} chunks")
    print(f"   Prefixo: search_document:")
    print()

    inserted = 0
    errors = 0
    start_time = time.time()

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = chunks[batch_start:batch_end]

        texts = [c.get("text", "") for c in batch_chunks]

        try:
            embeddings = get_embeddings_batch(texts, task_type="document")

            points = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk_id = chunk.get("chunk_id", "")
                if not chunk_id:
                    errors += 1
                    continue

                points.append(
                    PointStruct(
                        id=generate_qdrant_id(chunk_id),
                        vector=embedding.tolist(),
                        payload={
                            "chunk_id": chunk_id,
                            "document": chunk.get("document"),
                            "file_hash": chunk.get("file_hash"),
                            "page": chunk.get("page"),
                            "chunk": chunk.get("chunk"),
                            "text": chunk.get("text", ""),
                        },
                    )
                )

            if points:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )
                inserted += len(points)

            elapsed = time.time() - start_time
            rate = inserted / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / rate if rate > 0 else 0

            print(
                f"   📊 {batch_end}/{total} "
                f"({batch_end * 100 / total:.1f}%) "
                f"| {rate:.1f} chunks/s "
                f"| ETA: {eta:.0f}s"
            )

        except Exception as e:
            print(f"   ❌ Erro no lote {batch_num + 1}: {e}")
            errors += len(batch_chunks)

    elapsed = time.time() - start_time
    return inserted, errors, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Popula rag_documents_v2 com embeddings nomic-embed-text (Manuais + TSHOOT_KB)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recria a coleção do zero (apaga dados existentes)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa automaticamente o Passo 5 (Fine-Tuning incremental para novos documentos) após popular",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=8,
        help="Quantidade de chunks por documento no Passo 5 de treinamento (padrão: 8)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("📊 POPULANDO COLEÇÃO rag_documents_v2 (MANUAIS + RESOLUÇÕES TSHOOT)")
    print(f"   Modelo: {EMBEDDING_MODEL}")
    print(f"   Coleção: {QDRANT_COLLECTION}")
    print(f"   Qdrant: {QDRANT_URL}")
    print("=" * 70)

    print("\n🔌 Conectando ao Qdrant...")
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    create_collection(client, recreate=args.recreate)
    remove_synthetic_tshoot_kb(client)
    chunks = load_chunks()
    inserted, errors, elapsed = populate(client, chunks, incremental=not args.recreate)

    info = client.get_collection(QDRANT_COLLECTION)

    print()
    print("=" * 70)
    print("✅ POPULAÇÃO CONCLUÍDA!")
    print(f"   📊 Pontos inseridos: {inserted}")
    print(f"   ❌ Erros: {errors}")
    print(f"   ⏱️  Tempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"   📦 Total na coleção: {info.points_count}")
    print("=" * 70)

    data_dir = os.path.join(str(PROJECT_V2_ROOT), "data")
    os.makedirs(data_dir, exist_ok=True)

    manifest = {
        "collection": QDRANT_COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "source": CHUNKS_FILE,
        "points_inserted": inserted,
        "points_total": info.points_count,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(data_dir, "population_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Manifesto salvo em: {manifest_path}")

    train_script = Path(__file__).resolve().parent / "train_new_docs.py"
    if args.train:
        import subprocess
        print("\n🚀 Iniciando Etapa 5: Treinamento Incremental para Novos Documentos...")
        subprocess.run([
            sys.executable,
            str(train_script),
            "--chunks-per-doc",
            str(args.chunks_per_doc),
        ], check=False)
    else:
        print(
            "\n💡 Próximo passo (Etapa 5 - Treinar modelo com os novos documentos):\n"
            f"   python3 {train_script} --chunks-per-doc {args.chunks_per_doc}"
        )


if __name__ == "__main__":
    main()
