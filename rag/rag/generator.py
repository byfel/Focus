import logging

import requests

from .config import (
    OLLAMA_CHAT_URL,
    OLLAMA_CHAT_TIMEOUT,
    GENERATION_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    EVIDENCE_AUDIT_ENABLED,
)

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Erro ao gerar resposta com o LLM (modelo indisponível, OOM na
    GPU, resposta vazia, etc.)."""


SYSTEM_PROMPT = """
Você é um assistente técnico especializado
em sistemas de mídia, broadcast, áudio, vídeo,
infraestrutura e documentação técnica.

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
11. Não estabeleça relações causais entre informações
    apenas porque elas aparecem no mesmo contexto.

12. Diferencie claramente:
    - informação explicitamente documentada;
    - conclusão que pode ser deduzida diretamente;
    - hipótese ou possibilidade técnica.

13. Só apresente uma configuração como causa ou solução
    quando essa relação estiver explícita na documentação
    ou puder ser deduzida diretamente dela.

14. Quando a documentação mostrar dois parâmetros relacionados,
    mas não explicar uma relação causal entre eles, informe
    os dois como pontos de verificação sem afirmar que um
    necessariamente causa o outro.

15. Em troubleshooting, priorize os procedimentos e parâmetros
    explicitamente associados ao problema nos documentos.

16. Não inclua informações verdadeiras porém irrelevantes
    para responder à pergunta.
"""


def _build_context(results):
    """
    Organiza o contexto agrupando os chunks por documento.

    Isso reduz a chance de o modelo misturar informações
    provenientes de documentos diferentes.
    """

    documents = {}
    total_chars = 0
    dropped = 0

    for result in results:
        chunk = result["chunk"]

        document = chunk.get(
            "document",
            "documento_desconhecido"
        )

        if document not in documents:
            documents[document] = []

        documents[document].append(result)

    parts = []

    source_index = 1

    for document, document_results in documents.items():

        document_header = (
            f"\n"
            f"============================================================\n"
            f"DOCUMENTO: {document}\n"
            f"============================================================\n"
        )

        if total_chars + len(document_header) > MAX_CONTEXT_CHARS:
            dropped += len(document_results)
            continue

        parts.append(document_header)
        total_chars += len(document_header)

        for result in document_results:

            chunk = result["chunk"]

            piece = (
                f"\n--- FONTE {source_index} ---\n"
                f"Página: {chunk.get('page', '?')}\n"
                f"Chunk: {chunk.get('chunk', '?')}\n"
                f"Score: {result['score']:.4f}\n"
                f"Conteúdo:\n"
                f"{chunk.get('text', '')}\n"
            )

            if total_chars + len(piece) > MAX_CONTEXT_CHARS:
                dropped += 1
                continue

            parts.append(piece)
            total_chars += len(piece)

            source_index += 1

    if dropped:
        logger.warning(
            "Contexto truncado: %d fonte(s) descartada(s) "
            "por exceder MAX_CONTEXT_CHARS=%d.",
            dropped,
            MAX_CONTEXT_CHARS,
        )

    logger.debug(
        "Contexto final: %d documento(s), %d fonte(s), "
        "%d caracteres.",
        len(documents),
        source_index - 1,
        total_chars,
    )

    return "\n".join(parts)


def _audit_evidence(query: str, context: str, model: str) -> str:
    """
    Analisa o contexto recuperado antes da geração da resposta.

    O objetivo é separar:
    - fatos explicitamente documentados;
    - conclusões diretamente dedutíveis;
    - inferências que não são sustentadas pelo contexto.

    Pode levantar GenerationError — quem chama decide se quer
    fallback (ver generate_answer).
    """

    audit_prompt = f"""
Você é um auditor de evidências para um sistema RAG técnico.

PERGUNTA DO USUÁRIO:
{query}

CONTEXTO RECUPERADO:
{context}

Analise SOMENTE o que está presente no contexto.

Classifique as informações relevantes em três categorias:

1. FATOS_DOCUMENTADOS
Informações explicitamente afirmadas nos documentos.

2. CONCLUSOES_DIRETAS
Conclusões que podem ser obtidas diretamente a partir
das informações documentadas, sem adicionar conhecimento externo.

3. INFERENCIAS_NAO_COMPROVADAS
Relações de causa, substituição, dependência ou funcionamento
que NÃO estejam explicitamente documentadas e que não possam
ser deduzidas diretamente.

REGRAS:

- Não use conhecimento externo.
- Não complete lacunas com conhecimento técnico próprio.
- Não transforme duas informações relacionadas em uma relação
  causal sem evidência.
- Não considere uma configuração como causa de um comportamento
  apenas porque aparecem próximas no documento.
- Preserve documento e página quando disponíveis.
- Se uma informação não puder ser comprovada pelo contexto,
  coloque-a em INFERENCIAS_NAO_COMPROVADAS.

FORMATO:

FATOS_DOCUMENTADOS:
- ...

CONCLUSOES_DIRETAS:
- ...

INFERENCIAS_NAO_COMPROVADAS:
- ...
"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": audit_prompt,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                },
                "num_ctx": 32768,
            },
            timeout=OLLAMA_CHAT_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        audit = data.get("message", {}).get("content")

        if not audit:
            raise GenerationError(
                f"Auditoria de evidências retornou resposta vazia "
                f"para o modelo {model!r}."
            )

        return audit.strip()

    except requests.RequestException as exc:
        logger.error(
            "Falha na auditoria de evidências com o modelo %s.",
            model,
            exc_info=True,
        )
        raise GenerationError(
            f"Falha na auditoria de evidências: {exc}"
        ) from exc


def generate_answer(query: str, results: list, model: str) -> str:
    context = _build_context(results)

    # ========================================================
    # EVIDENCE AUDIT (com fallback)
    #
    # Se a auditoria falhar (Ollama fora do ar, timeout, OOM
    # trocando de modelo, etc.), não derrubamos a pergunta
    # inteira — seguimos para a geração normal, sem o filtro
    # extra, e avisamos tanto no log quanto na própria resposta
    # (de forma visível, não silenciosa).
    # ========================================================
    evidence_section = ""
    audit_skipped = False

    if EVIDENCE_AUDIT_ENABLED:
        logger.debug(
            "Evidence Audit habilitado para a pergunta: %r",
            query,
        )

        try:
            evidence = _audit_evidence(
                query=query,
                context=context,
                model=model,
            )

            logger.debug(
                "Evidence Audit concluído para: %r",
                query,
            )

            evidence_section = f"""
AUDITORIA DAS EVIDÊNCIAS:

{evidence}

Use esta auditoria como filtro de evidência.

IMPORTANTE:
- FATOS_DOCUMENTADOS podem ser utilizados.
- CONCLUSOES_DIRETAS podem ser utilizadas quando realmente
  forem consequência direta dos fatos.
- INFERENCIAS_NAO_COMPROVADAS NÃO devem ser apresentadas
  como fatos.
- Se a pergunta exigir uma relação que não esteja comprovada,
  informe claramente essa limitação.
"""

        except GenerationError:
            logger.warning(
                "Evidence Audit falhou para %r — seguindo sem "
                "auditoria (fallback).",
                query,
                exc_info=True,
            )
            audit_skipped = True
            evidence_section = ""

    user_prompt = f"""
CONTEXTO:
{context}

{evidence_section}

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
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "stream": False,
                "options": {
                    "temperature": GENERATION_TEMPERATURE,
                },
                "num_ctx": 32768,
            },
            timeout=OLLAMA_CHAT_TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as exc:
        logger.error(
            "Falha ao gerar resposta com o modelo %s.",
            model,
            exc_info=True,
        )
        raise GenerationError(
            f"Modelo {model!r} indisponível ou falhou ao responder: {exc}"
        ) from exc

    message = data.get("message", {}).get("content")

    if not message:
        raise GenerationError(
            f"Resposta vazia do modelo {model!r}."
        )

    answer = message.strip()

    if audit_skipped:
        answer = (
            "⚠️ Auditoria de evidências indisponível nesta resposta "
            "(seguiu sem essa checagem extra) — confira as fontes "
            "com atenção redobrada.\n\n" + answer
        )

    return answer