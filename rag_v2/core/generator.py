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
    
    system_prompt = """Você é um assistente técnico especializado em infraestrutura de broadcast, engenharia de TV e documentação técnica.

Sua tarefa é responder à pergunta do usuário utilizando as informações e instruções presentes no CONTEXTO fornecido.

DIRETRIZES:
1. Baseie sua resposta nas informações documentadas no contexto.
2. Se o contexto estiver em inglês, compreenda e traduza as explicações e procedimentos para português.
3. Preserve exatamente comandos, sintaxe de terminal, opções, nomes de arquivos e parâmetros de configuração.
4. Para distribuições Linux: reconheça que Rocky Linux e AlmaLinux são equivalentes a RHEL/CentOS. Se o contexto descrever a instalação para RHEL/CentOS, apresente os passos adequados indicando essa equivalência.
5. Se o contexto fornecido não contiver nenhuma informação sobre o assunto perguntado, responda estritamente: "Essa informação não foi encontrada na documentação fornecida."
6. Responda em português, de maneira técnica, clara e direta.
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
