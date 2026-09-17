#!/usr/bin/env python3
"""
Teste rápido do pipeline RAG V2 (nomic-embed-text).

Uso:
    python3 scripts/test_rag.py "Sua pergunta aqui"
    python3 scripts/test_rag.py   # pergunta padrão
"""

import sys
import time
from pathlib import Path

# Adiciona o path da V2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import ask
from core.config import EMBEDDING_MODEL, QDRANT_COLLECTION


def main():
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Como instalar o Autodesk Flame?"
    )

    print("=" * 60)
    print(f"🔵 RAG V2 — {EMBEDDING_MODEL} → {QDRANT_COLLECTION}")
    print(f"❓ {question}")
    print("=" * 60)

    start = time.time()

    try:
        result = ask(question)
        elapsed = time.time() - start

        print(f"\n📝 Resposta ({elapsed:.2f}s):\n")
        print(result["answer"])

        print(f"\n📚 Fontes ({len(result['sources'])}):")
        for i, source in enumerate(result["sources"], 1):
            print(
                f"  {i}. {source['document']} "
                f"(Pág. {source['page']}) "
                f"Score: {source['score']}"
            )

        print(f"\n⏱️  Tempo total: {elapsed:.2f}s")
        print(f"🤖 Modelo: {result.get('model', '?')}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Erro ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
