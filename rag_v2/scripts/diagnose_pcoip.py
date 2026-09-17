#!/usr/bin/env python3
"""
Script de diagnóstico comparativo V1 vs V2 para PCoIP.

Compara:
1. Coleção V1 (rag_documents) vs V2 (rag_documents_v2)
2. Total de pontos e lista de documentos em cada uma
3. Chunks que contêm 'pcoip', 'teradici' ou 'anyware'
4. Busca direta de teste em ambas as coleções
"""
import sys
from pathlib import Path

# Adiciona o path da V2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from core.config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL
from core.embeddings import get_embedding


def inspect_collection(client: QdrantClient, col_name: str):
    print(f"\n{'='*30} INSPEÇÃO: {col_name} {'='*30}")
    collections = [c.name for c in client.get_collections().collections]
    if col_name not in collections:
        print(f"❌ Coleção '{col_name}' NÃO EXISTE no Qdrant!")
        return {}

    info = client.get_collection(col_name)
    print(f"📦 Total de pontos: {info.points_count}")

    docs = {}
    pcoip_hits = []

    offset = None
    scanned = 0

    while True:
        records, offset = client.scroll(
            collection_name=col_name,
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
                pcoip_hits.append({
                    "doc": doc,
                    "page": payload.get("page"),
                    "snippet": (payload.get("text") or "")[:140].replace("\n", " ")
                })

        if offset is None:
            break

    print(f"📄 Documentos indexados ({len(docs)} arquivos):")
    for doc, count in sorted(docs.items(), key=lambda x: x[1], reverse=True):
        is_pcoip = any(k in doc.lower() for k in ["pcoip", "teradici", "anyware"])
        tag = " 🌟 [ARQUIVO PCoIP]" if is_pcoip else ""
        print(f"   • {doc}: {count} chunks{tag}")

    print(f"\n🔎 Chunks com termos PCoIP encontrados: {len(pcoip_hits)}")
    for h in pcoip_hits[:4]:
        print(f"   - {h['doc']} (pág. {h['page']}): \"{h['snippet']}...\"")

    return docs


def test_search(client: QdrantClient, col_name: str, query: str, model_type: str):
    print(f"\n🔍 Testando busca em '{col_name}' ({model_type})")
    print(f"   Query: \"{query}\"")

    try:
        if model_type == "v2_nomic":
            emb = get_embedding(query, task_type="query").tolist()
        else:
            # Em V1 usa embeddinggemma
            import requests
            res = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "embeddinggemma:latest", "input": query},
                timeout=30
            )
            emb = res.json()["embeddings"][0]

        res = client.query_points(
            collection_name=col_name,
            query=emb,
            limit=10,
            with_payload=True,
        )

        for rank, pt in enumerate(res.points, 1):
            p = pt.payload or {}
            doc = p.get("document", "?")
            page = p.get("page", "?")
            print(f"   {rank:2d}. Score: {pt.score:.4f} | {doc} (pág. {page})")

    except Exception as e:
        print(f"   ❌ Erro no teste de busca: {e}")


def main():
    print("=" * 80)
    print("🔬 DIAGNÓSTICO COMPARATIVO: V1 (rag_documents) vs V2 (rag_documents_v2)")
    print("=" * 80)

    client = QdrantClient(url=QDRANT_URL, timeout=30)

    # 1. Inspeciona V1
    docs_v1 = inspect_collection(client, "rag_documents")

    # 2. Inspeciona V2
    docs_v2 = inspect_collection(client, QDRANT_COLLECTION)

    # 3. Diferenças entre V1 e V2
    print(f"\n{'='*30} COMPARAÇÃO DE CONTEÚDO {'='*30}")
    missing_in_v2 = set(docs_v1.keys()) - set(docs_v2.keys())
    if missing_in_v2:
        print("⚠️ Documentos presentes na V1 que NÃO ESTÃO na V2:")
        for doc in missing_in_v2:
            print(f"   🚨 {doc} ({docs_v1[doc]} chunks na V1)")
    else:
        print("✅ Todos os documentos da V1 estão presentes na V2!")

    # 4. Testes práticos de busca
    q = "como configurar o Rocky Linux para que posso acessar ela via Pcoip / hp anywhre ?"
    test_search(client, "rag_documents", q, "v1_gemma")
    test_search(client, QDRANT_COLLECTION, q, "v2_nomic")

    # 5. Teste final do pipeline V2 com diversidade e brand boost
    print(f"\n{'='*30} TESTE DO PIPELINE V2 COM RETRIEVE() {'='*30}")
    from core.retriever import retrieve
    res_v2 = retrieve(q)
    print(f"✅ Contextos selecionados após Diversidade e Brand Boost ({len(res_v2)}):")
    for i, item in enumerate(res_v2, 1):
        c = item["chunk"]
        print(f"   {i:2d}. Score: {item['score']:.4f} | {c.get('document')} (pág. {c.get('page')})")

    print("\n" + "=" * 80)
    print("Fim do diagnóstico.")
    print("=" * 80)


if __name__ == "__main__":
    main()
