import requests
import numpy as np
from typing import List
from .config import EMBEDDING_MODEL, OLLAMA_EMBED_URL, CACHE_ENABLED

_embedding_cache = {}

def get_embedding(text: str) -> np.ndarray:
    """Obtém embedding com cache"""
    if CACHE_ENABLED and text in _embedding_cache:
        return _embedding_cache[text]
    
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=120
    )
    response.raise_for_status()
    
    embedding = np.array(response.json()["embeddings"][0], dtype=np.float32)
    
    if CACHE_ENABLED:
        _embedding_cache[text] = embedding
    
    return embedding

def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """Gera embeddings em lote"""
    results = []
    texts_to_process = []
    
    for text in texts:
        if CACHE_ENABLED and text in _embedding_cache:
            results.append(_embedding_cache[text])
        else:
            texts_to_process.append(text)
    
    if not texts_to_process:
        return results
    
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL, "input": texts_to_process},
            timeout=300
        )
        response.raise_for_status()
        
        for text, embedding in zip(texts_to_process, response.json()["embeddings"]):
            emb = np.array(embedding, dtype=np.float32)
            if CACHE_ENABLED:
                _embedding_cache[text] = emb
            results.append(emb)
        
        return results
        
    except Exception as e:
        print(f"❌ Batch embedding falhou: {e}")
        return [get_embedding(text) for text in texts_to_process]