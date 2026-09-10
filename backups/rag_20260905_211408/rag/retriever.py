import requests

from qdrant_client import QdrantClient

from .config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    TOP_K,
    FINAL_CONTEXTS,
    THRESHOLD,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_MODEL,
    QUERY_EXPANSION_COUNT,
    EXPANSION_TOP_K,
)

from .embeddings import get_embedding


# ============================================================
# QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL
)


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_CHAT_URL = (
    "http://localhost:11434/api/chat"
)


# ============================================================
# RRF
# ============================================================

RRF_K = 60


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

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "stream": False,

                "options": {
                    "temperature": 0.2
                }
            },

            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        content = (
            data
            .get("message", {})
            .get("content", "")
        )

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        cleaned = []

        for line in lines:

            line = line.lstrip(
                "0123456789.-) "
            ).strip()

            if line:
                cleaned.append(line)

        unique = []

        seen = set()

        for item in cleaned:

            normalized = (
                " ".join(
                    item.lower().split()
                )
            )

            if normalized in seen:

                continue

            seen.add(
                normalized
            )

            unique.append(
                item
            )

        expansions = unique[
            :QUERY_EXPANSION_COUNT
        ]

        return [
            query
        ] + expansions

    except Exception as e:

        print()

        print(
            "AVISO: falha no Query Expansion:"
        )

        print(e)

        print(
            "Continuando somente com a pergunta original."
        )

        return [
            query
        ]


# ============================================================
# BUSCA QDRANT
# ============================================================

def search_qdrant(
    query,
    top_k
):

    query_embedding = get_embedding(
        query
    )

    result = client.query_points(

        collection_name=QDRANT_COLLECTION,

        query=query_embedding,

        limit=top_k,

        with_payload=True,
    )

    candidates = []

    for rank, point in enumerate(
        result.points,
        start=1
    ):

        payload = point.payload or {}

        candidates.append({

            "id": point.id,

            "score": float(
                point.score
            ),

            "rank": rank,

            "document": payload.get(
                "document"
            ),

            "page": payload.get(
                "page"
            ),

            "chunk": payload.get(
                "chunk"
            ),

            "text": payload.get(
                "text",
                ""
            ),

        })

    return candidates


# ============================================================
# RRF SCORE
# ============================================================

def calculate_rrf_score(
    ranks
):

    return sum(
        1.0 / (
            RRF_K + rank
        )
        for rank in ranks
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    top_k=TOP_K,
    threshold=THRESHOLD
):

    # --------------------------------------------------------
    # QUERY EXPANSION
    # --------------------------------------------------------

    queries = expand_query(
        query
    )

    print()

    print("=" * 80)
    print("QUERY EXPANSION")
    print("=" * 80)

    print(
        f"Pergunta original: {query}"
    )

    for index, expanded in enumerate(
        queries[1:],
        start=1
    ):

        print(
            f"{index}. {expanded}"
        )

    print(
        f"Total de consultas: "
        f"{len(queries)}"
    )


    # --------------------------------------------------------
    # BUSCA TODAS AS CONSULTAS
    # --------------------------------------------------------

    all_candidates = {}

    print()

    print("=" * 80)
    print("BUSCA QDRANT")
    print("=" * 80)


    for query_index, current_query in enumerate(
        queries,
        start=1
    ):

        results = search_qdrant(

            current_query,

            EXPANSION_TOP_K
        )

        print()

        print(
            f"BUSCA {query_index}/"
            f"{len(queries)}"
        )

        print(
            f"Query: {current_query}"
        )

        print(
            f"Resultados: {len(results)}"
        )


        # ----------------------------------------------------
        # Guarda todos os resultados encontrados
        # ----------------------------------------------------

        for item in results:

            item_id = item["id"]

            if item_id not in all_candidates:

                all_candidates[
                    item_id
                ] = {

                    **item,

                    "rrf_ranks": [],

                    "appearances": 0,

                    "best_score": item[
                        "score"
                    ],

                    "best_rank": item[
                        "rank"
                    ],

                }


            candidate = all_candidates[
                item_id
            ]


            # ------------------------------------------------
            # Registra posição nessa consulta
            # ------------------------------------------------

            candidate[
                "rrf_ranks"
            ].append(
                item["rank"]
            )


            # ------------------------------------------------
            # Quantas vezes apareceu
            # ------------------------------------------------

            candidate[
                "appearances"
            ] += 1


            # ------------------------------------------------
            # Melhor score original
            # ------------------------------------------------

            if (
                item["score"]
                >
                candidate["best_score"]
            ):

                candidate[
                    "best_score"
                ] = item[
                    "score"
                ]


            # ------------------------------------------------
            # Melhor posição
            # ------------------------------------------------

            if (
                item["rank"]
                <
                candidate["best_rank"]
            ):

                candidate[
                    "best_rank"
                ] = item[
                    "rank"
                ]


    # --------------------------------------------------------
    # Calcula RRF
    # --------------------------------------------------------

    candidates = []

    for item in all_candidates.values():

        item["rrf_score"] = (
            calculate_rrf_score(
                item["rrf_ranks"]
            )
        )

        candidates.append(
            item
        )


    print()

    print("=" * 80)
    print("CANDIDATOS APÓS QUERY EXPANSION + RRF")
    print("=" * 80)


    # --------------------------------------------------------
    # Threshold
    #
    # Usa o melhor score original do Qdrant.
    # Não usa RRF para o threshold.
    # --------------------------------------------------------

    filtered = [

        item

        for item in candidates

        if item["best_score"]
        >= threshold

    ]


    if not filtered:

        print(
            "Nenhum candidato passou "
            f"do threshold {threshold:.4f}."
        )

        return []


    # --------------------------------------------------------
    # Ordenação principal: RRF
    #
    # Desempate:
    # 1. quantidade de aparições
    # 2. melhor score semântico
    # 3. melhor posição
    # --------------------------------------------------------

    filtered = sorted(

        filtered,

        key=lambda item: (

            item["rrf_score"],

            item["appearances"],

            item["best_score"],

            -item["best_rank"]

        ),

        reverse=True
    )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    ranked = filtered[
        :top_k
    ]


    # --------------------------------------------------------
    # DEBUG COMPLETO
    # --------------------------------------------------------

    for index, item in enumerate(
        ranked,
        start=1
    ):

        print(

            f"{index}. "

            f"RRF={item['rrf_score']:.6f} | "

            f"Score={item['best_score']:.4f} | "

            f"Aparições={item['appearances']} | "

            f"Ranks={item['rrf_ranks']} | "

            f"{item['document']} | "

            f"Página={item['page']} | "

            f"Chunk={item['chunk']}"

        )


    # ========================================================
    # SELEÇÃO DOS CONTEXTOS
    # ========================================================

    selected = ranked[
        :FINAL_CONTEXTS
    ]


    print()

    print("=" * 80)
    print("CONTEXTOS SELECIONADOS")
    print("=" * 80)


    for index, item in enumerate(
        selected,
        start=1
    ):

        print(

            f"{index}. "

            f"RRF={item['rrf_score']:.6f} | "

            f"Score={item['best_score']:.4f} | "

            f"{item['document']} | "

            f"Página={item['page']} | "

            f"Chunk={item['chunk']}"

        )


    print()


    # ========================================================
    # FORMATO PARA O GENERATOR
    # ========================================================

    return [

        {

            "score": item[
                "best_score"
            ],

            "chunk": {

                "document": item[
                    "document"
                ],

                "page": item[
                    "page"
                ],

                "chunk": item[
                    "chunk"
                ],

                "text": item[
                    "text"
                ],

            }

        }

        for item in selected

    ]