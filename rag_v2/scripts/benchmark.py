#!/usr/bin/env python3
"""
Compara performance entre V1 e V2
"""

import sys
import time
import json
import os

# Adiciona ambas versões
sys.path.insert(0, "/opt/ai/rag")
sys.path.insert(0, "/opt/ai/rag_v2")

# Perguntas de teste
QUESTIONS = [
    "O que é o Autodesk Flame?",
    "O que é PCoIP?",
    "Como configurar o MediaCentral?",
    "O que é Linux?",
    "Como editar vídeos no Media Composer?",
]

def test_version(name, ask_func):
    print(f"\n{'='*60}")
    print(f"📊 TESTANDO {name}")
    print(f"{'='*60}")
    
    results = []
    total_time = 0
    
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{i}. {q[:50]}...")
        
        start = time.time()
        try:
            r = ask_func(q)
            elapsed = time.time() - start
            total_time += elapsed
            
            results.append({
                "question": q,
                "time": elapsed,
                "sources": len(r.get("sources", [])),
                "answer_len": len(r.get("answer", ""))
            })
            
            print(f"   ✅ {elapsed:.2f}s | Fontes: {len(r.get('sources', []))}")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results.append({"question": q, "error": str(e)})
    
    avg = total_time / len(QUESTIONS)
    print(f"\n📊 Média: {avg:.2f}s")
    return {"avg": avg, "total": total_time, "results": results}

def main():
    print("=" * 60)
    print("🏆 BENCHMARK RAG V1 vs V2")
    print("=" * 60)
    
    # Testa V1
    try:
        from rag.pipeline import ask as ask_v1
        v1 = test_version("V1 (rag_documents)", ask_v1)
    except Exception as e:
        print(f"❌ Erro ao carregar V1: {e}")
        v1 = None
    
    # Testa V2
    try:
        from core.pipeline import ask as ask_v2
        v2 = test_version("V2 (rag_documents)", ask_v2)
    except Exception as e:
        print(f"❌ Erro ao carregar V2: {e}")
        v2 = None
    
    # Comparação
    if v1 and v2:
        print("\n" + "=" * 60)
        print("🏆 COMPARAÇÃO FINAL")
        print("=" * 60)
        print(f"V1: {v1['avg']:.2f}s")
        print(f"V2: {v2['avg']:.2f}s")
        
        improvement = ((v1['avg'] - v2['avg']) / v1['avg']) * 100
        print(f"📈 Melhoria: {improvement:.1f}%")
        
        # Salva resultados
        results = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "v1": v1,
            "v2": v2,
            "improvement": improvement
        }
        
        with open("/opt/ai/rag_v2/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Resultados salvos em: /opt/ai/rag_v2/benchmark_results.json")

if __name__ == "__main__":
    main()
