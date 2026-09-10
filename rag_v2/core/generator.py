import requests
from typing import List, Dict
from .config import OLLAMA_CHAT_URL, DEFAULT_LLM_MODEL

def build_context(results: List[Dict]) -> str:
    if not results:
        return ""
    
    parts = []
    for i, item in enumerate(results, 1):
        chunk = item.get("chunk", {})
        parts.append(
            f"""--- FONTE {i} ---
Documento: {chunk.get('document', 'desconhecido')}
Página: {chunk.get('page', '?')}
Score: {item.get('score', 0):.4f}

{chunk.get('text', '')}
"""
        )
    
    return "\n".join(parts)

def generate_answer(
    query: str,
    results: List[Dict],
    model: str = DEFAULT_LLM_MODEL
) -> str:
    if not results:
        return "Essa informação não foi encontrada na documentação fornecida."
    
    context = build_context(results)
    
    system_prompt = """Você é um assistente técnico especializado em documentação.

Responda APENAS com base no CONTEXTO fornecido.

REGRAS:
1. Não invente informações.
2. Preserve exatamente nomes de produtos e comandos.
3. Se a informação não estiver no contexto, responda: "Essa informação não foi encontrada na documentação fornecida."
4. Responda em português, de forma objetiva.
"""

    user_prompt = f"""CONTEXTO:

{context}

PERGUNTA: {query}

RESPOSTA:"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=600
        )
        response.raise_for_status()
        
        return response.json().get("message", {}).get("content", "").strip()
        
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return f"Erro ao gerar resposta: {str(e)}"
