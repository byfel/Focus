#!/usr/bin/env python3
"""
Benchmark comparativo: V1 (embeddinggemma) vs V2 (nomic-embed-text)

Compara tempo de resposta, qualidade das fontes e overlap de documentos
entre as duas versões do pipeline RAG.

Uso:
    cd /opt/ai/rag_v2
    python3 scripts/benchmark.py
"""

import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime

# Adiciona ambas versões ao path
sys.path.insert(0, "/opt/ai/rag")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# PERGUNTAS DE TESTE
# ============================================================

QUESTIONS = [
    "O que é o Autodesk Flame?",
    "O que é PCoIP?",
    "Como configurar o MediaCentral?",
    "O que é Linux?",
    "Como editar vídeos no Media Composer?",
    "O que é protocolo Dante?",
    "Como instalar o Flame no Linux?",
    "O que é uma Timeline no Media Composer?",
    "Quais são os requisitos de hardware do Flame?",
    "Como funciona o roteamento de áudio Dante?",
]


def test_version(name: str, ask_func, questions: list) -> dict:
    """Testa uma versão do pipeline com as perguntas de teste."""
    print(f"\n{'=' * 70}")
    print(f"📊 TESTANDO {name}")
    print(f"{'=' * 70}")

    results = []
    total_time = 0

    for i, q in enumerate(questions, 1):
        print(f"\n  {i}/{len(questions)}. {q}")

        start = time.time()
        try:
            r = ask_func(q)
            elapsed = time.time() - start
            total_time += elapsed

            sources = r.get("sources", [])
            source_docs = [s.get("document", "?") for s in sources[:5]]

            results.append({
                "question": q,
                "time": round(elapsed, 2),
                "sources_count": len(sources),
                "top_sources": source_docs[:3],
                "answer_length": len(r.get("answer", "")),
                "answer_preview": r.get("answer", "")[:150],
            })

            print(f"     ✅ {elapsed:.2f}s | {len(sources)} fontes")
            for j, doc in enumerate(source_docs[:3], 1):
                score = sources[j - 1].get("score", 0) if j <= len(sources) else 0
                print(f"        {j}. {doc} (score: {score:.4f})")

        except Exception as e:
            elapsed = time.time() - start
            print(f"     ❌ Erro ({elapsed:.2f}s): {e}")
            results.append({
                "question": q,
                "time": round(elapsed, 2),
                "error": str(e),
            })

    avg = total_time / len(questions) if questions else 0

    print(f"\n  📊 Resumo {name}:")
    print(f"     Tempo médio: {avg:.2f}s")
    print(f"     Tempo total: {total_time:.2f}s")

    return {
        "name": name,
        "avg_time": round(avg, 2),
        "total_time": round(total_time, 2),
        "results": results,
    }


def compare_results(v1: dict, v2: dict, questions: list):
    """Compara os resultados entre V1 e V2."""
    print("\n" + "=" * 70)
    print("🏆 COMPARAÇÃO V1 (embeddinggemma) vs V2 (nomic-embed-text)")
    print("=" * 70)

    # Tabela de tempos
    print(f"\n{'Pergunta':<45} {'V1':>7} {'V2':>7} {'Diff':>8}")
    print("-" * 70)

    for i, q in enumerate(questions):
        r1 = v1["results"][i] if i < len(v1["results"]) else {}
        r2 = v2["results"][i] if i < len(v2["results"]) else {}

        t1 = r1.get("time", 0)
        t2 = r2.get("time", 0)
        diff = t2 - t1
        indicator = "🟢" if diff < 0 else "🔴" if diff > 0 else "⚪"

        label = q[:43] + ".." if len(q) > 43 else q
        print(f"  {label:<43} {t1:>6.2f}s {t2:>6.2f}s {diff:>+7.2f}s {indicator}")

    print("-" * 70)
    print(
        f"  {'MÉDIA':<43} "
        f"{v1['avg_time']:>6.2f}s "
        f"{v2['avg_time']:>6.2f}s "
        f"{v2['avg_time'] - v1['avg_time']:>+7.2f}s"
    )

    # Comparação de fontes
    print(f"\n📚 Overlap de fontes (top 3 documentos):")
    print("-" * 70)

    for i, q in enumerate(questions):
        r1 = v1["results"][i] if i < len(v1["results"]) else {}
        r2 = v2["results"][i] if i < len(v2["results"]) else {}

        docs_v1 = set(r1.get("top_sources", []))
        docs_v2 = set(r2.get("top_sources", []))

        common = docs_v1 & docs_v2
        overlap = len(common) / max(len(docs_v1 | docs_v2), 1) * 100

        label = q[:35] + ".." if len(q) > 35 else q
        print(f"  {label:<37} overlap: {overlap:.0f}%  comuns: {common or '—'}")

    # Melhoria geral
    if v1["avg_time"] > 0:
        speed_change = ((v1["avg_time"] - v2["avg_time"]) / v1["avg_time"]) * 100
        print(f"\n📈 Velocidade: V2 é {abs(speed_change):.1f}% "
              f"{'mais rápida' if speed_change > 0 else 'mais lenta'} que V1")


def main():
    print("=" * 70)
    print("🏆 BENCHMARK RAG: V1 (embeddinggemma) vs V2 (nomic-embed-text)")
    print(f"   Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Perguntas: {len(QUESTIONS)}")
    print("=" * 70)

    # Testa V1
    v1 = None
    try:
        from rag.pipeline import ask as ask_v1
        v1 = test_version("V1 (embeddinggemma → rag_documents)", ask_v1, QUESTIONS)
    except Exception as e:
        print(f"\n❌ Erro ao carregar V1: {e}")
        import traceback
        traceback.print_exc()

    # Testa V2
    v2 = None
    try:
        from core.pipeline import ask as ask_v2
        v2 = test_version("V2 (nomic-embed-text → rag_documents_v2)", ask_v2, QUESTIONS)
    except Exception as e:
        print(f"\n❌ Erro ao carregar V2: {e}")
        import traceback
        traceback.print_exc()

    # Comparação
    if v1 and v2:
        compare_results(v1, v2, QUESTIONS)

        # Salva resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "questions_count": len(QUESTIONS),
            "v1": v1,
            "v2": v2,
        }

        results_dir = str(Path(__file__).resolve().parent.parent / "data")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"benchmark_{timestamp}.json")

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Resultados salvos em: {results_file}")

    elif v1:
        print("\n⚠️ Apenas V1 disponível. Popule a coleção V2 primeiro:")
        print("   python3 scripts/populate_v2.py")
    elif v2:
        print("\n⚠️ Apenas V2 disponível. V1 não pôde ser carregada.")
    else:
        print("\n❌ Nenhuma versão pôde ser carregada.")


if __name__ == "__main__":
    main()
