import requests
import numpy as np

from .config import (
    EMBEDDING_MODEL,
    OLLAMA_EMBED_URL
)


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
