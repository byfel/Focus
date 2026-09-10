import logging

import numpy as np
import requests

from .config import EMBEDDING_MODEL, OLLAMA_EMBED_URL, OLLAMA_EMBED_TIMEOUT

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Erro ao obter embedding do Ollama (serviço fora do ar, modelo
    não encontrado, resposta vazia, etc.)."""


def get_embedding(text: str) -> np.ndarray:
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=OLLAMA_EMBED_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Falha ao chamar o Ollama para embedding.", exc_info=True)
        raise EmbeddingError(f"Serviço de embedding indisponível: {exc}") from exc

    embeddings = data.get("embeddings")
    if not embeddings:
        raise EmbeddingError(
            f"Ollama não retornou embeddings para o texto "
            f"(modelo={EMBEDDING_MODEL!r})."
        )

    return np.array(embeddings[0], dtype=np.float32)
