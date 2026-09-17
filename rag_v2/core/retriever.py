import requests
from typing import List, Dict
from qdrant_client import QdrantClient
from .config import (
    QDRANT_URL, QDRANT_COLLECTION,
    TOP_K, FINAL_CONTEXTS, THRESHOLD,
    QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_COUNT,
    QUERY_EXPANSION_MODEL, OLLAMA_CHAT_URL
)
from .embeddings import get_embedding

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client

_expansion_cache = {}

def expand_query(query: str) -> List[str]:
    if not QUERY_EXPANSION_ENABLED:
        return [query]
    
    cache_key = f"expand_{query}"
    if cache_key in _expansion_cache:
        return _expansion_cache[cache_key]
    
    prompt = f"""Gere {QUERY_EXPANSION_COUNT} versões alternativas da pergunta abaixo.
Mantenha o significado técnico. Uma por linha.

Pergunta: {query}"""

    try:
        response = requests.post(
            f"{OLLAMA_CHAT_URL}",
            json={
                "model": QUERY_EXPANSION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=30
        )
        response.raise_for_status()
        
        content = response.json().get("message", {}).get("content", "")
        expansions = [line.strip() for line in content.splitlines() if line.strip()]
        expansions = expansions[:QUERY_EXPANSION_COUNT]
        
        result = [query] + expansions if expansions else [query]
        _expansion_cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"⚠️ Query expansion falhou: {e}")
        return [query]

def retrieve(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> List[Dict]:
    queries = expand_query(query)
    
    all_candidates = {}
    client = get_qdrant_client()
    
    for current_query in queries:
        query_embedding = get_embedding(current_query).tolist()
        
        try:
            result = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            continue
        
        for rank, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            item_id = point.id
            
            if item_id not in all_candidates:
                all_candidates[item_id] = {
                    "id": item_id,
                    "score": float(point.score),
                    "rank": rank,
                    "document": payload.get("document"),
                    "page": payload.get("page"),
                    "chunk": payload.get("chunk"),
                    "text": payload.get("text", ""),
                    "rrf_ranks": [],
                    "appearances": 0,
                    "best_score": float(point.score),
                }
            
            candidate = all_candidates[item_id]
            candidate["rrf_ranks"].append(rank)
            candidate["appearances"] += 1
            candidate["best_score"] = max(candidate["best_score"], float(point.score))
    
    RRF_K = 60
    for item in all_candidates.values():
        item["rrf_score"] = sum(1.0 / (RRF_K + rank) for rank in item["rrf_ranks"])
    
    candidates = [c for c in all_candidates.values() if c["best_score"] >= threshold]
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: (x["rrf_score"], x["appearances"], x["best_score"]), reverse=True)
    
    selected = candidates[:FINAL_CONTEXTS]
    
    return [
        {
            "score": item["best_score"],
            "chunk": {
                "document": item["document"],
                "page": item["page"],
                "chunk": item["chunk"],
                "text": item["text"],
            }
        }
        for item in selected
    ]
