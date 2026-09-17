#!/usr/bin/env python3
"""
Popula a coleção rag_documents_v2 com embeddings gerados pelo nomic-embed-text.

Lê os chunks.json (textos já extraídos dos PDFs pela V1), gera NOVOS
embeddings usando o modelo nomic-embed-text com prefixo "search_document:",
e insere na coleção rag_documents_v2 do Qdrant.

Uso:
    python3 scripts/populate_v2.py
    python3 scripts/populate_v2.py --recreate   # recria a coleção do zero
"""

import sys
import os
import json
import time
import hashlib
import argparse
from pathlib import Path

# Adiciona o path da V2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from core.config import (
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE,
    DATA_ROOT, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
)
from core.embeddings import get_embeddings_batch

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CHUNKS_FILE = os.path.join(DATA_ROOT, "database", "chunks.json")


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


def load_chunks() -> list:
    """Carrega chunks.json (textos extraídos dos PDFs pela V1)."""
    print(f"\n📂 Carregando chunks de: {CHUNKS_FILE}")

    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Arquivo não encontrado: {CHUNKS_FILE}")
        print(f"   Verifique se DATA_ROOT está correto: {DATA_ROOT}")
        sys.exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"   ✅ {len(chunks)} chunks carregados")

    # Mostra resumo por documento
    docs = {}
    for c in chunks:
        doc = c.get("document", "desconhecido")
        docs[doc] = docs.get(doc, 0) + 1

    print(f"   📄 {len(docs)} documentos:")
    for doc, count in sorted(docs.items()):
        print(f"      • {doc}: {count} chunks")

    return chunks


def populate(client: QdrantClient, chunks: list):
    """Gera embeddings com nomic-embed-text e insere no Qdrant."""
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

        # Extrai textos para embedding
        texts = [c.get("text", "") for c in batch_chunks]

        try:
            # Gera embeddings com prefixo search_document:
            embeddings = get_embeddings_batch(texts, task_type="document")

            # Monta os pontos para o Qdrant
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

            # Insere no Qdrant
            if points:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )
                inserted += len(points)

            # Progresso
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
        description="Popula rag_documents_v2 com embeddings nomic-embed-text"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recria a coleção do zero (apaga dados existentes)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("📊 POPULANDO COLEÇÃO rag_documents_v2")
    print(f"   Modelo: {EMBEDDING_MODEL}")
    print(f"   Coleção: {QDRANT_COLLECTION}")
    print(f"   Qdrant: {QDRANT_URL}")
    print("=" * 70)

    # 1. Conecta ao Qdrant
    print("\n🔌 Conectando ao Qdrant...")
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    # 2. Cria/verifica coleção
    create_collection(client, recreate=args.recreate)

    # 3. Carrega chunks
    chunks = load_chunks()

    # 4. Popula
    inserted, errors, elapsed = populate(client, chunks)

    # 5. Resultado final
    info = client.get_collection(QDRANT_COLLECTION)

    print()
    print("=" * 70)
    print("✅ POPULAÇÃO CONCLUÍDA!")
    print(f"   📊 Pontos inseridos: {inserted}")
    print(f"   ❌ Erros: {errors}")
    print(f"   ⏱️  Tempo total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"   📦 Total na coleção: {info.points_count}")
    print("=" * 70)

    # 6. Salva manifesto
    data_dir = os.path.join(
        str(Path(__file__).resolve().parent.parent), "data"
    )
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


if __name__ == "__main__":
    main()
