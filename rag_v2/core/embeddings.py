import requests
import numpy as np
from typing import List, Literal
from .config import (
    EMBEDDING_MODEL, OLLAMA_EMBED_URL,
    CACHE_ENABLED,
    NOMIC_SEARCH_DOC_PREFIX, NOMIC_SEARCH_QUERY_PREFIX,
)

_embedding_cache = {}

# Tipo de tarefa para prefixos assimétricos do nomic-embed-text
TaskType = Literal["query", "document"]


def _apply_prefix(text: str, task_type: TaskType) -> str:
    """Aplica o prefixo correto do nomic-embed-text conforme o tipo de tarefa.
    - "query"    → busca (search_query:)
    - "document" → indexação (search_document:)
    """
    if task_type == "document":
        return f"{NOMIC_SEARCH_DOC_PREFIX}{text}"
    return f"{NOMIC_SEARCH_QUERY_PREFIX}{text}"


def get_embedding(text: str, task_type: TaskType = "query") -> np.ndarray:
    """Gera embedding para um texto usando nomic-embed-text.
    
    Args:
        text: Texto para gerar embedding.
        task_type: "query" para busca, "document" para indexação.
    """
    cache_key = f"{task_type}:{text}"
    if CACHE_ENABLED and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    prefixed_text = _apply_prefix(text, task_type)
    
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": prefixed_text},
        timeout=120
    )
    response.raise_for_status()
    
    data = response.json()
    # /api/embed retorna {"embeddings": [[...]]}
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError(
            f"Ollama não retornou embeddings (modelo={EMBEDDING_MODEL})"
        )
    
    embedding = np.array(embeddings[0], dtype=np.float32)
    
    if CACHE_ENABLED:
        _embedding_cache[cache_key] = embedding
    
    return embedding


def get_embeddings_batch(
    texts: List[str], task_type: TaskType = "query"
) -> List[np.ndarray]:
    """Gera embeddings em lote usando nomic-embed-text.
    
    Args:
        texts: Lista de textos para gerar embeddings.
        task_type: "query" para busca, "document" para indexação.
    """
    results = []
    texts_to_process = []
    indices_to_process = []
    
    for i, text in enumerate(texts):
        cache_key = f"{task_type}:{text}"
        if CACHE_ENABLED and cache_key in _embedding_cache:
            results.append((i, _embedding_cache[cache_key]))
        else:
            texts_to_process.append(text)
            indices_to_process.append(i)
    
    if texts_to_process:
        prefixed_texts = [_apply_prefix(t, task_type) for t in texts_to_process]
        
        try:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBEDDING_MODEL, "input": prefixed_texts},
                timeout=600
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings_list = data.get("embeddings", [])
            
            for text, idx, embedding in zip(
                texts_to_process, indices_to_process, embeddings_list
            ):
                emb = np.array(embedding, dtype=np.float32)
                if CACHE_ENABLED:
                    cache_key = f"{task_type}:{text}"
                    _embedding_cache[cache_key] = emb
                results.append((idx, emb))
            
        except Exception as e:
            print(f"❌ Batch embedding falhou, tentando um a um: {e}")
            for text, idx in zip(texts_to_process, indices_to_process):
                emb = get_embedding(text, task_type)
                results.append((idx, emb))
    
    # Ordena pela posição original e retorna só os arrays
    results.sort(key=lambda x: x[0])
    return [emb for _, emb in results]
