import logging

import requests

from .config import (
    OLLAMA_CHAT_URL,
    OLLAMA_CHAT_TIMEOUT,
    GENERATION_TEMPERATURE,
    MAX_CONTEXT_CHARS,
)

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Erro ao gerar resposta com o LLM (modelo indisponível, OOM na
    GPU, resposta vazia, etc.)."""


SYSTEM_PROMPT = """
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


def _build_context(results):
    """Monta o bloco de contexto respeitando MAX_CONTEXT_CHARS.

    É uma trava por CARACTERES, não por tokens reais do modelo —
    Gemma e Qwen tokenizam de formas diferentes, então isso é uma
    aproximação conservadora, não uma contagem exata. Serve para
    evitar que o Ollama trunque o prompt silenciosamente sem
    ninguém perceber.
    """
    parts = []
    total_chars = 0
    dropped = 0

    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        piece = (
            f"\n--- FONTE {i} ---\n"
            f"Documento:\n{chunk.get('document', 'desconhecido')}\n"
            f"Página:\n{chunk.get('page', '?')}\n"
            f"Chunk:\n{chunk.get('chunk', '?')}\n"
            f"Score:\n{result['score']:.4f}\n"
            f"Conteúdo:\n{chunk.get('text', '')}\n"
        )

        if total_chars + len(piece) > MAX_CONTEXT_CHARS:
            dropped = len(results) - (i - 1)
            break

        parts.append(piece)
        total_chars += len(piece)

    if dropped:
        logger.warning(
            "Contexto truncado: %d de %d fonte(s) descartada(s) por "
            "exceder MAX_CONTEXT_CHARS=%d.",
            dropped, len(results), MAX_CONTEXT_CHARS,
        )

    return "\n".join(parts)


def generate_answer(query: str, results: list, model: str) -> str:
    context = _build_context(results)

    user_prompt = f"""
CONTEXTO:
{context}
---
PERGUNTA DO USUÁRIO:
{query}

RESPONDA:
"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": GENERATION_TEMPERATURE},
                "num_ctx": 32768,
                
            },
            timeout=OLLAMA_CHAT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Falha ao gerar resposta com o modelo %s.", model, exc_info=True)
        raise GenerationError(
            f"Modelo {model!r} indisponível ou falhou ao responder: {exc}"
        ) from exc

    message = data.get("message", {}).get("content")
    if not message:
        raise GenerationError(f"Resposta vazia do modelo {model!r}.")

    return message.strip()
