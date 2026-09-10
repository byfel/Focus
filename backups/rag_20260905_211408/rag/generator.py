import requests

from .config import OLLAMA_CHAT_URL


def generate_answer(
    query,
    results,
    model
):

    context_parts = []

    for i, result in enumerate(
        results,
        1
    ):

        chunk = result["chunk"]

        context_parts.append(
            f"""
--- FONTE {i} ---

Documento:
{chunk.get("document", "desconhecido")}

Página:
{chunk.get("page", "?")}

Chunk:
{chunk.get("chunk", "?")}

Score:
{result["score"]:.4f}

Conteúdo:

{chunk.get("text", "")}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
Você é um assistente técnico especializado
em Autodesk Flame, Linux, PCoIP e sistemas
de mídia.

Responda à pergunta utilizando SOMENTE as
informações presentes no CONTEXTO.

REGRAS OBRIGATÓRIAS:

1. Não invente informações.

2. Não crie comandos que não estejam no contexto.

3. Quando reproduzir um comando, preserve
   exatamente a sintaxe apresentada na documentação.

4. Se a informação não estiver no contexto,
   diga claramente:

   "Essa informação não foi encontrada
   nos documentos disponíveis."

5. Responda em português.

6. Seja objetivo e técnico.

7. Quando possível, informe documento e página.

8. Não use conhecimento externo para completar
   informações que não estejam presentes.

9. Não atribua uma informação a uma página
   diferente daquela apresentada no contexto.

10. Se o contexto não responder diretamente
    à pergunta, informe essa limitação em vez
    de tentar adivinhar.
"""

    user_prompt = f"""
CONTEXTO:

{context}

---

PERGUNTA DO USUÁRIO:

{query}

RESPONDA:
"""

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()