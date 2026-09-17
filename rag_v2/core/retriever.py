import re
import requests
from typing import List, Dict
from qdrant_client import QdrantClient
from .config import (
    QDRANT_URL, QDRANT_COLLECTION,
    TOP_K, FINAL_CONTEXTS, THRESHOLD,
    MAX_CHUNKS_PER_DOCUMENT,
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

# Termos de marca/software específicos para desambiguação técnica
BRAND_KEYWORDS = {
    "pcoip", "anyware", "anywhere", "teradici", "dante",
    "flame", "avid", "mediacentral", "rivage", "yamaha", "audinate"
}


def extract_technical_query(query: str) -> str:
    """Extrai termos técnicos principais da pergunta para uma busca focada."""
    # Normalização de erros de digitação comuns
    q_norm = re.sub(r'\banywhre\b', 'anywhere', query, flags=re.IGNORECASE)
    q_norm = re.sub(r'\banywere\b', 'anywhere', q_norm, flags=re.IGNORECASE)

    stopwords = {
        "como", "para", "posso", "pode", "qual", "quais", "onde", "quando", "quem",
        "esse", "essa", "este", "esta", "com", "sem", "por", "que", "uma", "uns",
        "das", "dos", "nas", "nos", "sobre", "qualquer", "mais", "menos", "acessar",
        "configurar", "instalar", "fazer", "existe", "são", "sao", "nele", "dela",
        "dele", "ela", "ele", "seus", "suas", "meu", "minha", "você", "voce",
        "procedimento", "processo", "comandos", "passos"
    }
    words = [w for w in re.findall(r'[a-zA-Z0-9_\-\.]{2,}', q_norm) if w.lower() not in stopwords]
    return " ".join(words) if words else query


def expand_query(query: str) -> List[str]:
    """Gera consultas complementares: a pergunta original + consulta técnica focada."""
    tech_query = extract_technical_query(query)
    base_queries = [query]
    if tech_query and tech_query.lower() != query.lower():
        base_queries.append(tech_query)

    if not QUERY_EXPANSION_ENABLED:
        return base_queries

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
        clean_expansions = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "•")):
                line = line[1:].strip()
            elif len(line) > 2 and line[0].isdigit() and line[1] in (".", ")", ":", "-"):
                line = line[2:].strip()
            if line.lower().startswith(("aqui estão", "aqui estao", "versão", "versao", "pergunta")):
                continue
            if line and line != query and len(line) > 5:
                clean_expansions.append(line)

        expansions = clean_expansions[:QUERY_EXPANSION_COUNT]
        result = base_queries + expansions if expansions else base_queries
        _expansion_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"⚠️ Query expansion LLM falhou: {e}")
        return base_queries


def retrieve(query: str, top_k: int = None, threshold: float = None) -> List[Dict]:
    top_k = top_k if top_k is not None else TOP_K
    threshold = threshold if threshold is not None else THRESHOLD
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

    # Identifica marcas/softwares específicos citados na pergunta
    query_lower = query.lower()
    target_brands = {b for b in BRAND_KEYWORDS if b in query_lower}

    # Extrai termos técnicos da pergunta
    stopwords = {"como", "para", "posso", "pode", "qual", "quais", "onde", "quando", "quem", "esse", "essa", "este", "esta", "com", "sem", "por", "que", "uma", "uns", "das", "dos", "nas", "nos", "sobre", "qualquer", "mais", "menos", "acessar", "configurar", "instalar", "fazer"}
    query_terms = [w for w in re.findall(r'[a-zA-Z0-9_\-\.]{3,}', query_lower) if w not in stopwords]

    RRF_K = 60
    for item in all_candidates.values():
        item["rrf_score"] = sum(1.0 / (RRF_K + rank) for rank in item["rrf_ranks"])
        doc_str = (item.get("document") or "").lower()
        text_str = (item.get("text") or "").lower()

        # Verifica se o chunk ou documento contém a marca/software perguntado
        item["brand_match"] = any(b in doc_str or b in text_str for b in target_brands) if target_brands else False
        item["keyword_hits"] = sum(1 for term in query_terms if term in doc_str or term in text_str)

    candidates = [c for c in all_candidates.values() if c["best_score"] >= threshold]

    if not candidates and all_candidates:
        best_possible = max(c["best_score"] for c in all_candidates.values())
        print(f"⚠️ Nenhum candidato atingiu threshold={threshold:.2f} para '{query}'. Melhor score: {best_possible:.4f}")
        if best_possible >= 0.20:
            print(f"   ℹ️ Ativando fallback com threshold tolerante (>= 0.20)")
            candidates = [c for c in all_candidates.values() if c["best_score"] >= 0.20]
        else:
            return []
    elif not candidates:
        return []

    # Ordenação prioritária:
    # 1. Correspondência direta com o software específico perguntado (ex: PCoIP, Dante, Flame)
    # 2. Quantidade de palavras-chave coincidentes
    # 3. Score RRF (combinação das buscas)
    # 4. Melhores scores brutos
    candidates.sort(
        key=lambda x: (
            x.get("brand_match", False),
            x["keyword_hits"] > 0,
            x["keyword_hits"],
            x["rrf_score"],
            x["appearances"],
            x["best_score"]
        ),
        reverse=True
    )

    # ========================================================
    # DIVERSIDADE DOCUMENTAL
    # Impede que livros enciclopédicos gerais (ex: Linux a Bíblia)
    # ocupem mais de 2 slots, abrindo espaço para manuais específicos.
    # ========================================================
    selected = []
    doc_counts = {}

    for item in candidates:
        doc = item.get("document") or "desconhecido"
        if doc_counts.get(doc, 0) < MAX_CHUNKS_PER_DOCUMENT:
            selected.append(item)
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
            if len(selected) >= FINAL_CONTEXTS:
                break

    # Completa até FINAL_CONTEXTS se sobrarem slots
    if len(selected) < FINAL_CONTEXTS:
        selected_ids = {s["id"] for s in selected}
        for item in candidates:
            if item["id"] not in selected_ids:
                selected.append(item)
                if len(selected) >= FINAL_CONTEXTS:
                    break

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
