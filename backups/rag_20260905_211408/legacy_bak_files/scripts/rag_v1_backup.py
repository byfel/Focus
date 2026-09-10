import json
import sys
import requests
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

EMBEDDING_MODEL = "embeddinggemma:latest"
LLM_MODEL = "gemma3:12b"

CHUNKS_FILE = "/opt/ai/rag/database/chunks.json"
EMBEDDINGS_FILE = "/opt/ai/rag/database/embeddings.json"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

TOP_K = 5
THRESHOLD = 0.35


# ============================================================
# CARREGAR BANCO
# ============================================================

def load_data():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    return chunks, embeddings


# ============================================================
# GERAR EMBEDDING DA PERGUNTA
# ============================================================

def get_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    return np.array(
        response.json()["embeddings"][0],
        dtype=np.float32
    )


# ============================================================
# SIMILARIDADE
# ============================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ============================================================
# RECUPERAÇÃO DOS CHUNKS
# ============================================================

def retrieve(
    query,
    top_k=TOP_K,
    threshold=THRESHOLD
):

    chunks, embeddings = load_data()

    query_embedding = get_embedding(query)

    results = []

    for i, embedding in enumerate(embeddings):

        if isinstance(embedding, list):

            vector = np.array(
                embedding,
                dtype=np.float32
            )

        elif isinstance(embedding, dict):

            vector = np.array(
                embedding["embedding"],
                dtype=np.float32
            )

        else:

            continue

        score = cosine_similarity(
            query_embedding,
            vector
        )

        results.append({
            "index": i,
            "score": float(score)
        })

    # Ordena do mais parecido para o menos parecido

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Aplica o threshold

    filtered_results = [
        r
        for r in results
        if r["score"] >= threshold
    ]

    # Retorna somente os TOP_K melhores

    return [
        {
            "chunk": chunks[r["index"]],
            "score": r["score"]
        }
        for r in filtered_results[:top_k]
    ]


# ============================================================
# GERAR RESPOSTA COM GEMMA
# ============================================================

def generate_answer(query, results):

    context_parts = []

    for i, result in enumerate(results, 1):

        chunk = result["chunk"]

        context_parts.append(
            f"""
--- FONTE {i} ---

Documento:
{chunk.get("document", "desconhecido")}

Página:
{chunk.get("page", "?")}

Chunk:
{chunk.get("chunk", "?")}

Score:
{result["score"]:.4f}

Conteúdo:

{chunk.get("text", "")}
"""
        )

    context = "\n".join(context_parts)

    # ========================================================
    # PROMPT DO SISTEMA
    # ========================================================

    system_prompt = """
Você é um assistente técnico especializado em Autodesk Flame.

Responda à pergunta do usuário usando PRINCIPALMENTE as informações
fornecidas no CONTEXTO.

Regras obrigatórias:

1. Não invente informações.

2. Não crie comandos que não estejam no contexto.

3. Se a informação solicitada não estiver no contexto,
   diga claramente que ela não foi encontrada nos documentos
   disponíveis.

4. Quando houver comandos, preserve exatamente a sintaxe
   apresentada no documento.

5. Seja objetivo e técnico.

6. Responda em português.

7. Quando possível, informe a página do documento usada como fonte.

8. Não use conhecimento externo para completar informações
   que não estejam presentes no CONTEXTO.

9. Se o contexto não responder diretamente à pergunta,
   informe essa limitação em vez de tentar adivinhar.

10. Diferencie claramente informações encontradas no documento
    de explicações suas.
"""

    # ========================================================
    # PROMPT DO USUÁRIO
    # ========================================================

    user_prompt = f"""
CONTEXTO:

{context}

---

PERGUNTA DO USUÁRIO:

{query}
"""

    # ========================================================
    # CHAMADA AO OLLAMA
    # ========================================================

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Verifica se foi informada uma pergunta
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            'Uso: python rag.py "sua pergunta"'
        )

        sys.exit(1)

    query = " ".join(sys.argv[1:])


    # --------------------------------------------------------
    # Busca
    # --------------------------------------------------------

    print()
    print("Buscando informações...")
    print()


    results = retrieve(
        query,
        top_k=TOP_K,
        threshold=THRESHOLD
    )


    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print("Chunks recuperados:")

    if not results:

        print("Nenhum chunk atingiu o threshold.")

        print()
        print("=" * 80)
        print("RESPOSTA")
        print("=" * 80)
        print()

        print(
            "Não encontrei essa informação nos documentos disponíveis."
        )

        sys.exit(0)


    # --------------------------------------------------------
    # Mostrar chunks encontrados
    # --------------------------------------------------------

    for i, result in enumerate(results, 1):

        chunk = result["chunk"]

        print(
            f"{i}. "
            f"Página {chunk.get('page', '?')} "
            f"| Chunk {chunk.get('chunk', '?')} "
            f"| Score {result['score']:.4f}"
        )


    # --------------------------------------------------------
    # Gerar resposta
    # --------------------------------------------------------

    print()
    print("Gerando resposta com Gemma...")
    print()


    answer = generate_answer(
        query,
        results
    )


    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print("=" * 80)
    print("RESPOSTA")
    print("=" * 80)
    print()

    print(answer)

