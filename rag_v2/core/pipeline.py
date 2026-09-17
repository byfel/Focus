from typing import Dict, List, Optional
from .retriever import retrieve
from .generator import generate_answer
from .config import DEFAULT_LLM_MODEL

class RAGPipeline:
    def __init__(self, model: str = DEFAULT_LLM_MODEL):
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
                "chunk": r["chunk"].get("chunk"),
                "score": round(r["score"], 4),
                "text": r["chunk"].get("text", ""),
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

def ask(question: str, model: str = DEFAULT_LLM_MODEL) -> Dict:
    pipeline = RAGPipeline(model)
    return pipeline.query(question)
