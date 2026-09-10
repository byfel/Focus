import logging

from .config import LLM_MODELS as SUPPORTED_MODELS, DEFAULT_LLM_MODEL as DEFAULT_MODEL
from .retriever import retrieve
from .generator import generate_answer, GenerationError
from .embeddings import EmbeddingError

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Essa informação não foi encontrada na documentação fornecida."


def ask(query: str, model: str = DEFAULT_MODEL) -> dict:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo não suportado: {model}")

    # retrieve() já protege internamente contra falha de embedding
    # e de Qdrant (retorna [] e loga), então um erro aqui indica algo
    # realmente inesperado — deixamos subir para o server.py registrar
    # e responder com 503.
    try:
        results = retrieve(query)
    except Exception:
        logger.exception(
            "Falha inesperada na etapa de retrieval para a pergunta %r", query
        )
        raise

    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    try:
        answer = generate_answer(query, results, model)
    except (GenerationError, EmbeddingError) as exc:
        logger.error("Falha ao gerar resposta para %r: %s", query, exc)
        raise

    sources = [
        {
            "document": result["chunk"].get("document"),
            "page": result["chunk"].get("page"),
            "chunk": result["chunk"].get("chunk"),
            "score": round(result["score"], 4),
        }
        for result in results
    ]

    return {"answer": answer, "sources": sources}
