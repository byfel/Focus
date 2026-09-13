import logging

import requests
from qdrant_client import QdrantClient

from .config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    OLLAMA_CHAT_URL,
    TOP_K,
    FINAL_CONTEXTS,
    THRESHOLD,
    RRF_K,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_COUNT,
    QUERY_EXPANSION_TEMPERATURE,
    EXPANSION_TOP_K,
    DOCUMENT_DIVERSITY_ENABLED,
    MAX_CHUNKS_PER_DOCUMENT,
)
from .embeddings import get_embedding, EmbeddingError

logger = logging.getLogger(__name__)

client = QdrantClient(url=QDRANT_URL)


# ============================================================
# QUERY EXPANSION
# ============================================================
def expand_query(query):
    if not QUERY_EXPANSION_ENABLED:
        return [query]

    prompt = f"""
Você é um especialista em recuperação de informação
para documentação técnica.

A pergunta original do usuário é:

{query}

Gere exatamente {QUERY_EXPANSION_COUNT} versões alternativas
da pergunta para melhorar uma busca semântica em documentação.

REGRAS:

1. Preserve exatamente o significado da pergunta.
2. Preserve nomes de produtos, softwares, equipamentos,
   fabricantes, comandos e tecnologias mencionados.
3. Não invente produtos, equipamentos, comandos ou conceitos.
4. Use palavras e expressões alternativas que possam aparecer
   em manuais técnicos.
5. Quando fizer sentido, utilize termos técnicos equivalentes
   ou sinônimos usados em documentação.
6. Não responda à pergunta.
7. Gere somente as perguntas alternativas.
8. Uma pergunta por linha.
9. Não use numeração.
10. Evite gerar perguntas que sejam apenas pequenas
    mudanças gramaticais da pergunta original.

Responda somente com as perguntas alternativas.
"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": QUERY_EXPANSION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": QUERY_EXPANSION_TEMPERATURE},
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("message", {}).get("content", "")

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        cleaned = []

        for line in lines:
            line = line.lstrip("0123456789.-) ").strip()

            if line:
                cleaned.append(line)

        unique = []
        seen = set()

        for item in cleaned:
            normalized = " ".join(item.lower().split())

            if normalized in seen:
                continue

            seen.add(normalized)
            unique.append(item)

        expansions = unique[:QUERY_EXPANSION_COUNT]

        return [query] + expansions

    except Exception:
        logger.warning(
            "Falha no Query Expansion, continuando somente com "
            "a pergunta original.",
            exc_info=True,
        )

        return [query]


# ============================================================
# BUSCA QDRANT
# ============================================================
def search_qdrant(query, top_k):
    try:
        query_embedding = get_embedding(query)

    except EmbeddingError:
        logger.error(
            "Não foi possível gerar embedding para a query %r.",
            query,
            exc_info=True,
        )
        return []

    try:
        result = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

    except Exception:
        logger.error(
            "Falha na consulta ao Qdrant para %r.",
            query,
            exc_info=True,
        )
        return []

    candidates = []

    for rank, point in enumerate(result.points, start=1):
        payload = point.payload or {}

        candidates.append(
            {
                "id": point.id,
                "score": float(point.score),
                "rank": rank,
                "document": payload.get("document"),
                "page": payload.get("page"),
                "chunk": payload.get("chunk"),
                "text": payload.get("text", ""),
            }
        )

    return candidates


def calculate_rrf_score(ranks):
    return sum(
        1.0 / (RRF_K + rank)
        for rank in ranks
    )


# ============================================================
# DOCUMENT DIVERSITY
# ============================================================
def select_diverse_contexts(candidates, final_contexts):
    """
    Seleciona os melhores candidatos preservando diversidade
    entre documentos.

    A seleção acontece em duas passagens:

    1. Garante representação inicial dos documentos relevantes.
    2. Preenche o restante respeitando MAX_CHUNKS_PER_DOCUMENT.

    A ordenação RRF continua sendo a principal referência.
    """

    if not candidates:
        return []

    if not DOCUMENT_DIVERSITY_ENABLED:
        return candidates[:final_contexts]

    selected = []
    selected_ids = set()
    document_counts = {}

    # --------------------------------------------------------
    # PASSO 1
    # Seleciona no máximo um chunk por documento.
    #
    # Isso evita que um único documento ocupe imediatamente
    # todo o contexto em perguntas que dependem de múltiplas
    # fontes.
    # --------------------------------------------------------
    for item in candidates:
        document = item.get("document") or "DESCONHECIDO"

        if document in document_counts:
            continue

        selected.append(item)
        selected_ids.add(item["id"])
        document_counts[document] = 1

        if len(selected) >= final_contexts:
            break

    # --------------------------------------------------------
    # PASSO 2
    # Completa os slots restantes respeitando o limite
    # máximo por documento.
    # --------------------------------------------------------
    for item in candidates:
        if len(selected) >= final_contexts:
            break

        if item["id"] in selected_ids:
            continue

        document = item.get("document") or "DESCONHECIDO"

        current_count = document_counts.get(document, 0)

        if current_count >= MAX_CHUNKS_PER_DOCUMENT:
            continue

        selected.append(item)
        selected_ids.add(item["id"])

        document_counts[document] = current_count + 1

    logger.debug(
        "Diversidade documental: %d contexto(s) selecionado(s) "
        "de %d documento(s). Distribuição=%s",
        len(selected),
        len(document_counts),
        document_counts,
    )

    return selected


# ============================================================
# RETRIEVE
# ============================================================
def retrieve(query, top_k=TOP_K, threshold=THRESHOLD):
    queries = expand_query(query)

    logger.debug(
        "Query expansion: %d consulta(s) para %r (%s)",
        len(queries),
        query,
        queries[1:],
    )

    all_candidates = {}

    # --------------------------------------------------------
    # Executa as buscas
    # --------------------------------------------------------
    for current_query in queries:
        results = search_qdrant(
            current_query,
            EXPANSION_TOP_K,
        )

        logger.debug(
            "Busca Qdrant %r -> %d resultado(s)",
            current_query,
            len(results),
        )

        for item in results:
            item_id = item["id"]

            if item_id not in all_candidates:
                all_candidates[item_id] = {
                    **item,
                    "rrf_ranks": [],
                    "appearances": 0,
                    "best_score": item["score"],
                    "best_rank": item["rank"],
                }

            candidate = all_candidates[item_id]

            candidate["rrf_ranks"].append(item["rank"])
            candidate["appearances"] += 1

            if item["score"] > candidate["best_score"]:
                candidate["best_score"] = item["score"]

            if item["rank"] < candidate["best_rank"]:
                candidate["best_rank"] = item["rank"]

    candidates = list(all_candidates.values())

    # --------------------------------------------------------
    # Calcula RRF
    # --------------------------------------------------------
    for item in candidates:
        item["rrf_score"] = calculate_rrf_score(
            item["rrf_ranks"]
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------
    filtered = [
        item
        for item in candidates
        if item["best_score"] >= threshold
    ]

    if not filtered:
        logger.info(
            "Nenhum candidato passou do threshold %.4f para %r",
            threshold,
            query,
        )
        return []

    # --------------------------------------------------------
    # Ordenação principal continua sendo RRF.
    # --------------------------------------------------------
    filtered = sorted(
        filtered,
        key=lambda item: (
            item["rrf_score"],
            item["appearances"],
            item["best_score"],
            -item["best_rank"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # TOP_K candidatos para seleção final
    # --------------------------------------------------------
    ranked = filtered[:top_k]

    # --------------------------------------------------------
    # Seleção com diversidade documental
    # --------------------------------------------------------
    selected = select_diverse_contexts(
        ranked,
        FINAL_CONTEXTS,
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------
    for index, item in enumerate(selected, start=1):
        logger.debug(
            "%d. RRF=%.6f Score=%.4f Aparições=%d "
            "%s p.%s chunk=%s",
            index,
            item["rrf_score"],
            item["best_score"],
            item["appearances"],
            item["document"],
            item["page"],
            item["chunk"],
        )

    return [
        {
            "score": item["best_score"],
            "chunk": {
                "document": item["document"],
                "page": item["page"],
                "chunk": item["chunk"],
                "text": item["text"],
            },
        }
        for item in selected
    ]