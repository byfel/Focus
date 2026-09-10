#!/usr/bin/env python3
import sys
import os

# Adiciona o path da V2
sys.path.insert(0, "/opt/ai/rag_v2")

from core.pipeline import ask

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Como instalar o Autodesk Flame?"
    
    print("=" * 60)
    print(f"❓ {question}")
    print("=" * 60)
    
    try:
        result = ask(question)
        
        print(f"\n📝 Resposta:\n{result['answer']}")
        print(f"\n📚 Fontes: {len(result['sources'])}")
        for source in result['sources']:
            print(f"  - {source['document']} (Pág. {source['page']}) Score: {source['score']}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
