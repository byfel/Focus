import logging
import requests

from .config import (
    OLLAMA_CHAT_URL,
    OLLAMA_CHAT_TIMEOUT,
    GENERATION_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class TShootError(Exception):
    """Erro durante a análise de troubleshooting."""


TSHOOT_SYSTEM_PROMPT = """
Você é um especialista em troubleshooting técnico
para sistemas de mídia, broadcast, áudio, vídeo,
infraestrutura e equipamentos profissionais.

Sua função é analisar um problema técnico utilizando
SOMENTE as evidências fornecidas.

Você NÃO deve utilizar conhecimento externo para
preencher lacunas.

Diferencie rigorosamente:

1. FATOS_DOCUMENTADOS
Informações explicitamente presentes na documentação.

2. CAUSAS_DOCUMENTADAS
Causas que a documentação relaciona explicitamente
ao sintoma ou comportamento apresentado.

3. CAUSAS_DEDUZIVEIS
Conclusões que podem ser obtidas diretamente das
evidências, sem adicionar conhecimento externo.

4. HIPOTESES
Possibilidades técnicas que podem explicar o problema,
mas que NÃO estão comprovadas pela documentação.

Hipóteses devem ser apresentadas explicitamente como
hipóteses e nunca como diagnóstico confirmado.

5. COMO_CONFIRMAR
Procedimentos de verificação que podem confirmar
ou descartar as hipóteses.

Somente apresente procedimentos que estejam
documentados ou que sejam consequência direta
das evidências fornecidas.

6. ACAO_RECOMENDADA
Ações sustentadas pela documentação.

Não invente comandos, menus, parâmetros,
configurações ou procedimentos.

7. LIMITACOES
Explique claramente o que a documentação não permite
determinar.

8. FONTES
Informe documento e página sempre que essas
informações estiverem disponíveis.

REGRAS:

- Não invente informações.
- Não transforme correlação em causalidade.
- Não trate hipótese como fato.
- Não use conhecimento externo.
- Não crie comandos.
- Não crie procedimentos inexistentes na documentação.
- Se não houver evidência suficiente, diga isso claramente.
- Responda em português.
- Seja técnico e objetivo.

FORMATO OBRIGATÓRIO:

SINTOMA:
...

FATOS_DOCUMENTADOS:
- ...

CAUSAS_DOCUMENTADAS:
- ...

CAUSAS_DEDUZIVEIS:
- ...

HIPOTESES:
- ...

COMO_CONFIRMAR:
1. ...
2. ...

ACAO_RECOMENDADA:
1. ...
2. ...

LIMITACOES:
- ...

FONTES:
- Documento: ...
  Página: ...
"""


def run_tshoot(
    query: str,
    context: str,
    evidence_audit: str = "",
    model: str = None,
) -> str:
    """
    Executa uma análise de troubleshooting baseada
    exclusivamente nas evidências recuperadas.

    Parâmetros:
        query:
            Pergunta/sintoma apresentado pelo usuário.

        context:
            Contexto recuperado pelo RAG.

        evidence_audit:
            Resultado opcional do Evidence Audit.

        model:
            Modelo Ollama utilizado para a análise.
    """

    if not model:
        from .config import DEFAULT_LLM_MODEL
        model = DEFAULT_LLM_MODEL

    prompt = f"""
PERGUNTA / SINTOMA DO USUÁRIO:

{query}


CONTEXTO DOCUMENTAL:

{context}
"""

    if evidence_audit:
        prompt += f"""

AUDITORIA DAS EVIDÊNCIAS:

{evidence_audit}
"""

    prompt += """

Analise o problema utilizando somente as informações
acima.

Não tente completar informações ausentes com conhecimento
externo.

Produza o diagnóstico seguindo exatamente a estrutura
definida pelo sistema.
"""

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": TSHOOT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
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
            "Falha no T-Shoot com o modelo %s.",
            model,
            exc_info=True,
        )

        raise TShootError(
            f"Falha no T-Shoot: {exc}"
        ) from exc

    message = data.get("message", {}).get("content")

    if not message:
        raise TShootError(
            f"T-Shoot retornou resposta vazia "
            f"para o modelo {model!r}."
        )

    return message.strip()