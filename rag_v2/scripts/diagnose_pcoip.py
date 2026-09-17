#!/usr/bin/env python3
"""
Script de diagnóstico para verificar por que perguntas sobre PCoIP não estão trazendo resultados.

Verifica:
1. Quantidade total de pontos em rag_documents_v2
2. Quais documentos (PDFs) estão de fato indexados na coleção
3. Se existem chunks contendo 'pcoip', 'teradici' ou 'anyware'
4. Scores brutos do Qdrant para as perguntas problemáticas
"""
import sys
import os
from pathlib import Path

# Adiciona o path da V2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from core.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL, THRESHOLD
from core.embeddings import get_embedding
from core.retriever import retrieve


def main():
    print("=" * 75)
    print("🔍 DIAGNÓSTICO DE BUSCA RAG v2 — PCoIP & DOCUMENTOS INDEXADOS")
    print(f"   Coleção: {QDRANT_COLLECTION}")
    print(f"   Qdrant: {QDRANT_URL}")
    print(f"   Modelo de Embedding: {EMBEDDING_MODEL}")
    print(f"   Threshold configurado: {THRESHOLD}")
    print("=" * 75)

    client = QdrantClient(url=QDRANT_URL, timeout=30)

    # 1. Informações da Coleção
    try:
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            print(f"\n❌ Coleção '{QDRANT_COLLECTION}' NÃO EXISTE no Qdrant!")
            print("   Execute: python3 scripts/populate_v2.py")
            return

        info = client.get_collection(QDRANT_COLLECTION)
        print(f"\n📦 Coleção encontrada!")
        print(f"   Total de pontos indexados: {info.points_count}")
    except Exception as e:
        print(f"\n❌ Erro ao conectar ao Qdrant: {e}")
        return

    # 2. Scroll de amostra para identificar documentos presentes
    print("\n📚 Identificando documentos indexados na coleção...")
    docs = {}
    pcoip_chunks_found = 0
    pcoip_sample = []

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
            doc = payload.get("document", "desconhecido")
            docs[doc] = docs.get(doc, 0) + 1

            text = (payload.get("text") or "").lower()
            if "pcoip" in text or "teradici" in text or "anyware" in text:
                pcoip_chunks_found += 1
                if len(pcoip_sample) < 3:
                    pcoip_sample.append({
                        "doc": doc,
                        "page": payload.get("page"),
                        "text": payload.get("text", "")[:180],
                    })

        if offset is None:
            break

    print(f"   Total escaneado: {scanned} chunks")
    print(f"   📄 {len(docs)} documentos únicos encontrados:")
    for doc, count in sorted(docs.items(), key=lambda x: x[1], reverse=True):
        is_pcoip = "pcoip" in doc.lower() or "teradici" in doc.lower()
        tag = " 🌟 [PCoIP]" if is_pcoip else ""
        print(f"      • {doc}: {count} chunks{tag}")

    print(f"\n🔎 Chunks que mencionam 'pcoip' / 'teradici' / 'anyware': {pcoip_chunks_found}")
    if pcoip_sample:
        print("   Exemplos encontrados no banco:")
        for s in pcoip_sample:
            print(f"      - {s['doc']} (pág. {s['page']}): \"{s['text']}...\"")

    # 3. Teste das perguntas problemáticas
    test_queries = [
        "como configurar e solucionar problemas de latência no protocolo PCoIP?",
        "como instalar o Pcoip no Rocky linux ?",
        "PCoIP latency troubleshooting",
        "install PCoIP Rocky Linux CentOS",
    ]

    print("\n" + "=" * 75)
    print("🎯 TESTE DE RECUPERAÇÃO (RETRIEVAL) DIRETO")
    print("=" * 75)

    for q in test_queries:
        print(f"\n❓ Pergunta: \"{q}\"")
        try:
            # Busca direta no Qdrant
            emb = get_embedding(q, task_type="query").tolist()
            qdrant_res = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=emb,
                limit=5,
                with_payload=True,
            )

            print("   Top 5 no Qdrant (scores brutos):")
            for rank, pt in enumerate(qdrant_res.points, 1):
                p = pt.payload or {}
                doc = p.get("document", "?")
                page = p.get("page", "?")
                score = pt.score
                status = "✅ PASSOU" if score >= THRESHOLD else "❌ REJEITADO PELO THRESHOLD"
                print(f"      {rank}. Score: {score:.4f} ({status}) | {doc} (pág. {page})")

            # Busca via pipeline retriever
            pipeline_res = retrieve(q)
            print(f"   -> Retorno do retrieve(): {len(pipeline_res)} contexto(s)")

        except Exception as e:
            print(f"   ❌ Erro ao buscar: {e}")

    print("\n" + "=" * 75)
    print("💡 CONCLUSÃO DO DIAGNÓSTICO:")
    if pcoip_chunks_found == 0:
        print("   🚨 NENHUM chunk sobre PCoIP foi encontrado no banco!")
        print("      Causa: Os PDFs de PCoIP não foram incluídos no chunks.json ou")
        print("      o populate_v2.py foi interrompido antes de chegar neles.")
    else:
        print(f"   ✅ Existem {pcoip_chunks_found} chunks sobre PCoIP na coleção.")
        print("      Se o score bruto estiver abaixo do threshold (ex: < 0.30),")
        print("      o ajuste do threshold resolverá a busca.")
    print("=" * 75)


if __name__ == "__main__":
    main()
