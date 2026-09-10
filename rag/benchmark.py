import os
import sys
import time
import statistics


# ============================================================
# CAMINHO DO PROJETO
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTAR RAG
# ============================================================

from rag.pipeline import ask
from rag.config import DEFAULT_LLM_MODEL as DEFAULT_MODEL


# ============================================================
# CONFIGURAÇÃO
# ============================================================

QUESTION = (
    "O que é o Dante Controller e para que ele é "
    "utilizado em uma rede Dante?"
)

NUM_TESTS = 5


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():

    print()
    print("=" * 70)
    print("BENCHMARK DO RAG")
    print("=" * 70)
    print()

    print(f"Modelo: {DEFAULT_MODEL}")
    print(f"Testes: {NUM_TESTS}")
    print()
    print("Pergunta:")
    print(QUESTION)
    print()

    results = []

    for i in range(1, NUM_TESTS + 1):

        print("-" * 70)
        print(f"TESTE {i}/{NUM_TESTS}")
        print("-" * 70)

        start = time.perf_counter()

        result = ask(
            query=QUESTION,
            model=DEFAULT_MODEL
        )

        end = time.perf_counter()

        total_time = end - start

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # ----------------------------------------------------
        # GUARDAR RESULTADO
        # ----------------------------------------------------

        results.append({
            "time": total_time,
            "sources": len(sources),
            "answer_length": len(answer)
        })

        print(f"Tempo total:       {total_time:.3f} s")
        print(f"Fontes:            {len(sources)}")
        print(f"Tamanho resposta:  {len(answer)} caracteres")

        print()
        print("Resposta:")
        print(answer)
        print()

        # ----------------------------------------------------
        # MOSTRAR FONTES
        # ----------------------------------------------------

        if sources:

            print("Fontes utilizadas:")

            for source in sources:

                document = source.get("document")
                page = source.get("page")
                chunk = source.get("chunk")
                score = source.get("score")

                print(
                    f"  - {document} | "
                    f"página {page} | "
                    f"chunk {chunk} | "
                    f"score {score}"
                )

            print()

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    times = [
        item["time"]
        for item in results
    ]

    print()
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print()

    print(
        f"Tempo médio:       {statistics.mean(times):.3f} s"
    )

    print(
        f"Tempo mínimo:      {min(times):.3f} s"
    )

    print(
        f"Tempo máximo:      {max(times):.3f} s"
    )

    print(
        f"Desvio padrão:     "
        f"{statistics.stdev(times):.3f} s"
        if len(times) > 1
        else "Desvio padrão:     N/A"
    )

    print()

    # ========================================================
    # TESTES INDIVIDUAIS
    # ========================================================

    print("=" * 70)
    print("TESTES INDIVIDUAIS")
    print("=" * 70)
    print()

    for i, item in enumerate(results, start=1):

        print(
            f"Teste {i}: "
            f"{item['time']:.3f} s | "
            f"{item['sources']} fontes | "
            f"{item['answer_length']} caracteres"
        )

    print()
    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    run_benchmark()