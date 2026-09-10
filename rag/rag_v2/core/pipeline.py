from typing import Dict, List
from .retriever import retrieve
from .generator import generate_answer
from .config import DEFAULT_LLM_MODEL, SUPPORTED_MODELS

class RAGPipeline:
    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Modelo não suportado: {model}")
        self.model = model
        self._cache = {}
        self._stats = {"total_queries": 0, "cache_hits": 0}
    
    def query(self, question: str, top_k: int = None, threshold: float = None) -> Dict:
        cache_key = f"{question}_{top_k}_{threshold}"
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        self._stats["total_queries"] += 1
        
        results = retrieve(question, top_k=top_k, threshold=threshold)
        answer = generate_answer(question, results, self.model)
        
        sources = [
            {
                "document": r["chunk"]["document"],
                "page": r["chunk"]["page"],
                "score": round(r["score"], 4)
            }
            for r in results
        ]
        
        result = {
            "answer": answer,
            "sources": sources,
            "contexts_used": len(results),
            "model": self.model
        }
        
        self._cache[cache_key] = result
        return result
    
    def get_stats(self) -> Dict:
        return {**self._stats, "cache_size": len(self._cache)}

def ask(question: str, model: str = DEFAULT_LLM_MODEL) -> Dict:
    pipeline = RAGPipeline(model)
    return pipeline.query(question)
