import json
import sys
import requests
import numpy as np

EMBEDDING_MODEL = "embeddinggemma:latest"

CHUNKS_FILE = "/opt/ai/rag/database/chunks.json"
EMBEDDINGS_FILE = "/opt/ai/rag/database/embeddings.json"

OLLAMA_URL = "http://localhost:11434/api/embed"


def load_data():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    return chunks, embeddings


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.array(data["embeddings"][0], dtype=np.float32)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def search(query, top_k=5):

    chunks, embeddings = load_data()

    query_embedding = get_embedding(query)

    results = []

    for i, embedding in enumerate(embeddings):

        # Caso o JSON contenha diretamente o vetor
        if isinstance(embedding, list):
            vector = np.array(embedding, dtype=np.float32)

        # Caso o JSON tenha {"embedding": [...]}
        elif isinstance(embedding, dict):
            vector = np.array(
                embedding["embedding"],
                dtype=np.float32
            )

        else:
            continue

        score = cosine_similarity(query_embedding, vector)

        results.append({
            "index": i,
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print()
    print("=" * 80)
    print("PERGUNTA")
    print("=" * 80)
    print(query)

    print()
    print("=" * 80)
    print(f"TOP {top_k} RESULTADOS")
    print("=" * 80)

    for rank, result in enumerate(results[:top_k], 1):

        index = result["index"]
        score = result["score"]

        chunk = chunks[index]

        print()
        print(f"[{rank}] Score: {score:.4f}")
        print(f"Página: {chunk.get('page', '?')}")
        print(f"Chunk: {chunk.get('chunk', '?')}")
        print("-" * 80)
        print(chunk.get("text", "")[:1500])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Uso: python search.py "sua pergunta aqui"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    search(query)
