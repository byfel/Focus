import os
import sys
import argparse
import requests


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from rag.retriever import retrieve
from rag.config import (
    DEFAULT_LLM_MODEL,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

OLLAMA_URL = os.getenv(
    "RAG_OLLAMA_URL",
    "http://localhost:11434"
)

TIMEOUT = int(
    os.getenv(
        "RAG_TIMEOUT",
        "900"
    )
)


# ============================================================
# MODELOS DISPONÍVEIS
# ============================================================

SUPPORTED_MODELS = [
    "gemma4:26b-a4b-it-q4_K_M",
    "qwen3.6:27b",
    "qwen3.6:35b",
    "gemma4:12b",
    "gemma4:e4b",
    "gemma3:12b",
    "gemma3:4b",
]


# ============================================================
# MODELO PADRÃO
# ============================================================

DEFAULT_MODEL = "gemma4:12b"


# ============================================================
# MODELO PARA CONTEXTO CONVERSACIONAL
# ============================================================

CONVERSATION_MODEL = "gemma4:e4b"

MAX_CONVERSATION_MESSAGES = 12


# ============================================================
# CONTEXTO
# ============================================================

def build_context(contexts):
    """
    Converte os resultados do retriever em texto
    para envio ao LLM.
    """

    if not contexts:
        return ""


    parts = []


    for index, item in enumerate(
        contexts,
        start=1
    ):

        chunk = item.get(
            "chunk",
            {}
        )

        document = chunk.get(
            "document",
            "desconhecido"
        )

        page = chunk.get(
            "page",
            "?"
        )

        chunk_number = chunk.get(
            "chunk",
            "?"
        )

        text = chunk.get(
            "text",
            ""
        )


        parts.append(

            f"""
--- CONTEXTO {index} ---

Documento: {document}
Página: {page}
Chunk: {chunk_number}

{text}
"""

        )


    return "\n".join(
        parts
    )


# ============================================================
# REESCRITA CONTEXTUAL DA PERGUNTA
# ============================================================

def rewrite_question(
    question,
    history,
    debug=False
):
    """
    Reescreve a pergunta atual utilizando
    o histórico da conversa.

    Exemplo:

    Histórico:
        Qual é a função do receptor Sennheiser?

    Pergunta:
        E onde ele fica?

    Resultado:
        Onde o receptor Sennheiser fica instalado?
    """

    # --------------------------------------------------------
    # Sem histórico não há necessidade de reescrever
    # --------------------------------------------------------

    if not history:

        return question


    # --------------------------------------------------------
    # Montar histórico
    # --------------------------------------------------------

    history_text = ""


    for message in history:

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            ""
        )


        if role == "user":

            history_text += (
                f"Usuário: {content}\n"
            )


        elif role == "assistant":

            history_text += (
                f"Assistente: {content}\n"
            )


    # --------------------------------------------------------
    # Prompt de reescrita
    # --------------------------------------------------------

    prompt = f"""
Você é um auxiliar de um sistema de busca
em documentação técnica.

Sua tarefa é reescrever a PERGUNTA ATUAL
para que ela possa ser pesquisada em uma
base de documentação.

Use o HISTÓRICO somente para resolver
referências e contexto.

REGRAS:

1. Preserve exatamente nomes de produtos,
equipamentos e softwares quando eles aparecerem
no histórico.

2. Resolva referências como:
"ele", "ela", "isso", "esse equipamento",
"esse sistema", "onde fica", "como faço",
"qual cabo", "qual comando", etc.

3. Não responda à pergunta.

4. Não invente informações.

5. Retorne somente uma única pergunta
reescrita.

6. Se a pergunta já estiver clara,
mantenha seu significado original.

7. Não adicione informações que não estejam
presentes na pergunta ou no histórico.

============================================================
HISTÓRICO
============================================================

{history_text}

============================================================
PERGUNTA ATUAL
============================================================

{question}

============================================================
PERGUNTA REESCRITA
============================================================
"""


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if debug:

        print()
        print("=" * 80)
        print("REESCRITA CONTEXTUAL")
        print("=" * 80)
        print()
        print(
            f"Modelo: {CONVERSATION_MODEL}"
        )
        print(
            f"Pergunta original: {question}"
        )
        print()


    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    try:

        response = requests.post(

            f"{OLLAMA_URL}/api/generate",

            json={

                "model":
                    CONVERSATION_MODEL,

                "prompt":
                    prompt,

                "stream":
                    False,

                "options": {

                    "temperature":
                        0.0

                }

            },

            timeout=300

        )

        response.raise_for_status()


    except requests.exceptions.RequestException as e:

        if debug:

            print(
                "AVISO: falha na reescrita contextual:"
            )

            print(e)

            print(
                "Continuando com a pergunta original."
            )

            print()

        return question


    data = response.json()


    rewritten = data.get(
        "response",
        ""
    ).strip()


    # --------------------------------------------------------
    # Validar resultado
    # --------------------------------------------------------

    if not rewritten:

        if debug:

            print(
                "AVISO: modelo não retornou "
                "uma pergunta reescrita."
            )

            print(
                "Continuando com a pergunta original."
            )

            print()

        return question


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if debug:

        print(
            f"Pergunta reescrita: {rewritten}"
        )

        print()


    return rewritten


# ============================================================
# GERAR RESPOSTA
# ============================================================

def generate_answer(
    question,
    context,
    llm_model,
    debug=False
):

    if not context:

        return (
            "Essa informação não foi encontrada "
            "na documentação fornecida."
        )


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
Você é um assistente técnico especializado
em documentação de sistemas de mídia,
Autodesk Flame, Linux, PCoIP,
Avid, MediaCentral, Media Composer,
equipamentos de áudio, vídeo e infraestrutura.

Responda sempre em português.

Sua resposta deve ser baseada SOMENTE
nas informações presentes no CONTEXTO.

REGRAS:

1. Não invente informações.

2. Não use conhecimento externo.

3. Não faça suposições técnicas que não estejam
   explicitamente ou claramente presentes no contexto.

4. Preserve exatamente nomes de equipamentos,
   produtos, softwares, comandos e opções.

5. Se a informação solicitada não estiver presente
   no contexto, responda exatamente:

   "Essa informação não foi encontrada na documentação fornecida."

6. Seja objetivo e técnico.

7. Responda diretamente à pergunta.

8. Quando a informação estiver claramente identificada,
   informe o documento e a página correspondente.

9. Nunca invente uma página.

10. Se houver informações de mais de um documento,
    utilize somente aquelas relevantes para a pergunta.

11. Não mencione o processo de RAG,
    embeddings, Qdrant, ranking ou recuperação.

12. Não diga que "o contexto informa".
    Responda diretamente.

13. Se houver uma resposta direta no contexto,
    prefira essa informação em vez de informações
    genéricas ou indiretas.

14. Não misture equipamentos diferentes apenas porque
    aparecem próximos no contexto.

15. Caso existam informações conflitantes,
    deixe isso explícito em vez de escolher uma
    informação arbitrariamente.
"""


    prompt = f"""
{system_prompt}

============================================================
CONTEXTO
============================================================

{context}

============================================================
PERGUNTA
============================================================

{question}

============================================================
RESPOSTA
============================================================
"""


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    if debug:

        print()
        print("=" * 80)
        print("DIAGNÓSTICO DO CONTEXTO")
        print("=" * 80)

        print(
            f"Caracteres do contexto: "
            f"{len(context)}"
        )

        print(
            f"Caracteres do prompt: "
            f"{len(prompt)}"
        )

        print()

        print("=" * 80)
        print("GERANDO RESPOSTA")
        print("=" * 80)

        print()

        print(
            f"Modelo LLM: "
            f"{llm_model}"
        )

        print(
            f"Tamanho do contexto: "
            f"{len(context)} caracteres"
        )

        print(
            f"Tamanho do prompt: "
            f"{len(prompt)} caracteres"
        )

        print()

        print("DEBUG")
        print("-" * 80)

        print(
            f"OLLAMA URL: "
            f"{OLLAMA_URL}"
        )

        print(
            f"Modelo: "
            f"{llm_model}"
        )

        print(
            f"Timeout: "
            f"{TIMEOUT} segundos"
        )

        print("-" * 80)


    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    try:

        response = requests.post(

            f"{OLLAMA_URL}/api/generate",

            json={

                "model":
                    llm_model,

                "prompt":
                    prompt,

                "stream":
                    False,

                "options": {

                    "temperature":
                        0.1

                }

            },

            timeout=TIMEOUT

        )

        response.raise_for_status()


    except requests.exceptions.RequestException as e:

        print()

        print("=" * 80)
        print("ERRO DE COMUNICAÇÃO COM OLLAMA")
        print("=" * 80)

        print()

        print(e)

        print()

        return None


    data = response.json()


    # --------------------------------------------------------
    # DEBUG OLLAMA
    # --------------------------------------------------------

    if debug:

        print(
            "Resposta recebida do Ollama."
        )

        print(
            f"done: "
            f"{data.get('done')}"
        )

        print(
            f"done_reason: "
            f"{data.get('done_reason')}"
        )


        if data.get(
            "total_duration"
        ):

            print(

                f"total_duration: "

                f"{data['total_duration'] / 1e9:.2f}s"

            )


        if data.get(
            "prompt_eval_count"
        ):

            print(

                f"prompt_eval_count: "

                f"{data['prompt_eval_count']}"

            )


        if data.get(
            "eval_count"
        ):

            print(

                f"eval_count: "

                f"{data['eval_count']}"

            )


        print()


    return data.get(
        "response",
        ""
    ).strip()


# ============================================================
# RAG
# ============================================================

def ask(
    question,
    llm_model,
    history=None,
    debug=False
):

    if history is None:

        history = []


    if llm_model not in SUPPORTED_MODELS:

        raise ValueError(
            f"Modelo não suportado: "
            f"{llm_model}"
        )


    print(
        f"LLM: {llm_model}"
    )

    print(
        "Retriever: rag.retriever.retrieve()"
    )

    print()


    # ========================================================
    # PERGUNTA CONTEXTUAL
    # ========================================================

    search_question = rewrite_question(

        question,

        history,

        debug=debug

    )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    contexts = retrieve(

        search_question

    )


    # ========================================================
    # DIAGNÓSTICO DOS CONTEXTOS
    # ========================================================

    if debug:

        print("=" * 80)
        print("DIAGNÓSTICO DOS CONTEXTOS")
        print("=" * 80)

        print(
            f"Pergunta usada no retrieval: "
            f"{search_question}"
        )

        print()

        print(
            f"Contextos recuperados: "
            f"{len(contexts)}"
        )

        print()


        for index, item in enumerate(
            contexts,
            start=1
        ):

            chunk = item.get(
                "chunk",
                {}
            )


            print(

                f"{index}. "

                f"Score="
                f"{item.get('score', 0):.4f} | "

                f"{chunk.get('document', 'desconhecido')} | "

                f"Página="
                f"{chunk.get('page', '?')} | "

                f"Chunk="
                f"{chunk.get('chunk', '?')}"

            )

        print()


    # ========================================================
    # CONTEXTO
    # ========================================================

    context = build_context(

        contexts

    )


    # ========================================================
    # RESPOSTA
    # ========================================================

    answer = generate_answer(

        question=
            question,

        context=
            context,

        llm_model=
            llm_model,

        debug=
            debug

    )


    return {

        "question":
            question,

        "search_question":
            search_question,

        "answer":
            answer,

        "model":
            llm_model,

        "contexts":
            contexts,

        "context":
            context,

    }


# ============================================================
# CHAT CONVERSACIONAL
# ============================================================

def chat(
    llm_model,
    debug=False
):

    history = []


    print()
    print("=" * 80)
    print("RAG CONVERSACIONAL")
    print("=" * 80)
    print()

    print(
        f"Modelo de resposta: {llm_model}"
    )

    print(
        f"Modelo de contexto: {CONVERSATION_MODEL}"
    )

    print()

    print(
        "Digite 'sair' para encerrar."
    )

    print()


    while True:

        try:

            question = input(
                "Você: "
            ).strip()


        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()

            break


        if not question:

            continue


        if question.lower() in (

            "sair",
            "exit",
            "quit"

        ):

            print()

            break


        # ====================================================
        # EXECUTAR RAG
        # ====================================================

        result = ask(

            question=
                question,

            llm_model=
                llm_model,

            history=
                history,

            debug=
                debug

        )


        answer = result.get(
            "answer"
        )


        # ====================================================
        # EXIBIR RESPOSTA
        # ====================================================

        print()

        print(
            "RAG:"
        )

        print(
            answer
        )

        print()


        # ====================================================
        # SALVAR HISTÓRICO
        # ====================================================

        history.append({

            "role":
                "user",

            "content":
                question

        })


        history.append({

            "role":
                "assistant",

            "content":
                answer

        })


        # ====================================================
        # LIMITAR HISTÓRICO
        # ====================================================

        if len(history) > MAX_CONVERSATION_MESSAGES:

            history = history[
                -MAX_CONVERSATION_MESSAGES:
            ]


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=
            "RAG com Ollama + Qdrant"

    )


    parser.add_argument(

        "question",

        nargs="*",

        help=
            "Pergunta para o RAG"

    )


    parser.add_argument(

        "--model",

        "-m",

        default=
            DEFAULT_MODEL,

        choices=
            SUPPORTED_MODELS,

        help=(

            "Modelo LLM a utilizar "

            f"(padrão: {DEFAULT_MODEL})"

        )

    )


    parser.add_argument(

        "--chat",

        action=
            "store_true",

        help=
            "Inicia uma conversa contínua com memória"

    )


    parser.add_argument(

        "--debug",

        action=
            "store_true",

        help=
            "Exibe informações de diagnóstico"

    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()


    # ========================================================
    # MODO CHAT
    # ========================================================

    if args.chat:

        chat(

            llm_model=
                args.model,

            debug=
                args.debug

        )

        return


    # ========================================================
    # MODO PERGUNTA ÚNICA
    # ========================================================

    if not args.question:

        print(
            "Erro: informe uma pergunta "
            "ou utilize --chat."
        )

        sys.exit(1)


    question = " ".join(
        args.question
    )


    try:

        result = ask(

            question=
                question,

            llm_model=
                args.model,

            debug=
                args.debug

        )


    except Exception as e:

        print()

        print("=" * 80)
        print("ERRO")
        print("=" * 80)

        print()

        print(
            type(e).__name__
        )

        print(
            e
        )

        print()

        sys.exit(1)


    if result[
        "answer"
    ] is None:

        sys.exit(1)


    print("=" * 80)
    print("RESPOSTA")
    print("=" * 80)

    print()

    print(
        result["answer"]
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()
