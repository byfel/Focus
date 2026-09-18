#!/usr/bin/env python3
"""
Script de Diagnóstico Semântico: Variações da Consulta Kaspersky.

Testa as 5 consultas sugeridas para avaliar o comportamento do embedding e da busca:
A) Instalação do Antivírus Kaspersky
B) Procedimento de instalação do Kaspersky
C) Procedimento de instalação do Kaspersky no Linux
D) Kaspersky Linux
E) Kaspersky instalação

Compara:
1. Busca Vetorial Pura no Qdrant (sem pré/pós-processamento)
2. Pipeline Completo V2 com retrieve() (Multi-Query, Entity Boost e Diversidade)
"""

import sys
import os
from pathlib import Path

# Adiciona a raiz do rag_v2 ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from core.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from core.embeddings import get_embedding
from core.retriever import retrieve, analyze_query_components, generate_search_queries


QUERIES = [
    ("A", "Instalação do Antivírus Kaspersky"),
    ("B", "Procedimento de instalação do Kaspersky"),
    ("C", "Procedimento de instalação do Kaspersky no Linux"),
    ("D", "Kaspersky Linux"),
    ("E", "Kaspersky instalação"),
]


def test_raw_vector_search(client: QdrantClient, query: str, limit: int = 10):
    """Executa busca vetorial pura no Qdrant usando nomic-embed-text."""
    emb = get_embedding(query, task_type="query").tolist()
    try:
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=emb,
            limit=limit,
            with_payload=True,
        )
        return results.points
    except Exception as e:
        print(f"   ❌ Erro na busca vetorial: {e}")
        return []


def print_separator(char="=", length=85):
    print(char * length)


def main():
    print_separator("=")
    print("🔬 DIAGNÓSTICO SEMÂNTICO: AVALIAÇÃO DE ENTIDADE (KASPERSKY)")
    print(f"   Coleção: {QDRANT_COLLECTION}")
    print(f"   Modelo de Embedding: {EMBEDDING_MODEL}")
    print(f"   Qdrant: {QDRANT_URL}")
    print_separator("=")

    client = QdrantClient(url=QDRANT_URL, timeout=30)

    # 1. Verifica se existem chunks de Kaspersky na coleção
    print("\n🔍 Verificando presença de termos 'kaspersky' na base...")
    kaspersky_chunks = []
    offset = None
    scanned = 0

    while True:
        records, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for r in records:
            scanned += 1
            payload = r.payload or {}
            doc = payload.get("document", "")
            text = payload.get("text", "")
            if "kaspersky" in doc.lower() or "kaspersky" in text.lower():
                kaspersky_chunks.append({
                    "id": r.id,
                    "doc": doc,
                    "page": payload.get("page"),
                    "chunk": payload.get("chunk"),
                    "snippet": text[:120].replace("\n", " ")
                })

        if offset is None:
            break

    print(f"   Total de pontos escaneados: {scanned}")
    print(f"   🎯 Chunks contendo 'kaspersky' encontrados: {len(kaspersky_chunks)}")
    for kc in kaspersky_chunks[:5]:
        print(f"      • {kc['doc']} (pág. {kc['page']}, chunk {kc['chunk']}): \"{kc['snippet']}...\"")

    if not kaspersky_chunks:
        print("   ⚠️ ATENÇÃO: Nenhum chunk com o termo 'kaspersky' foi encontrado na coleção!")
        print("      Verifique se o manual do Kaspersky foi adicionado e ingerido.")

    # 2. Testa cada uma das 5 variações
    for letter, query in QUERIES:
        print("\n" + "=" * 85)
        print(f"📌 TESTE {letter}) \"{query}\"")
        print("=" * 85)

        # Decomposição semântica
        comp = analyze_query_components(query)
        generated_queries = generate_search_queries(query)
        print(f"   🧠 Análise Semântica:")
        print(f"      - Entidades: {comp['entities']}")
        print(f"      - Plataformas: {comp['platforms']}")
        print(f"      - Ações: {comp['actions']}")
        print(f"      - Consultas geradas ({len(generated_queries)}): {generated_queries}")

        # A) Busca Vetorial Pura (Raw Qdrant)
        print(f"\n   [1] 📊 Busca Vetorial Pura (Top 10):")
        raw_points = test_raw_vector_search(client, query, limit=10)
        if raw_points:
            for rank, pt in enumerate(raw_points, 1):
                p = pt.payload or {}
                doc = p.get("document", "?")
                page = p.get("page", "?")
                text = (p.get("text") or "").lower()
                is_kaspersky = "kaspersky" in doc.lower() or "kaspersky" in text
                tag = " 🎯 [KASPERSKY MATCH]" if is_kaspersky else ""
                print(f"       {rank:2d}. Score: {pt.score:.4f} | {doc} (pág. {page}){tag}")
        else:
            print("       Nenhum resultado retornado.")

        # B) Pipeline Completo retrieve()
        print(f"\n   [2] 🛡️ Pipeline Completo V2 retrieve() (Top Contextos com Boost & Diversidade):")
        pipeline_results = retrieve(query)
        if pipeline_results:
            for rank, item in enumerate(pipeline_results, 1):
                c = item["chunk"]
                doc = c.get("document", "?")
                page = c.get("page", "?")
                text = (c.get("text") or "").lower()
                is_kaspersky = "kaspersky" in doc.lower() or "kaspersky" in text
                tag = " 🎯 [KASPERSKY MATCH]" if is_kaspersky else ""
                print(f"       {rank:2d}. Score: {item['score']:.4f} | {doc} (pág. {page}){tag}")
        else:
            print("       Nenhum contexto selecionado.")

    print("\n" + "=" * 85)
    print("✅ DIAGNÓSTICO CONCLUÍDO!")
    print("=" * 85)


if __name__ == "__main__":
    main()
