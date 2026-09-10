import sys
sys.path.insert(0, "/opt/ai/rag_v2")

from core.pipeline import ask

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Como instalar o Autodesk Flame?"
    
    print("=" * 60)
    print(f"❓ {question}")
    print("=" * 60)
    
    result = ask(question)
    
    print(f"\n📝 Resposta:\n{result['answer']}")
    print(f"\n📚 Fontes: {len(result['sources'])}")
    for source in result['sources']:
        print(f"  - {source['document']} (Pág. {source['page']}) Score: {source['score']}")

if __name__ == "__main__":
    main()