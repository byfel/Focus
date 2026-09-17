import json
import re
import hashlib
import base64
import requests
import argparse

from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DOCUMENTS_DIR = Path("/opt/ai/rag/documents")
DATABASE_DIR = Path("/opt/ai/rag/database")

OUTPUT_FILE = DATABASE_DIR / "chunks.json"
STATE_FILE = DATABASE_DIR / "ingest_state.json"

CHUNK_SIZE = 1500

MIN_VALID_CHUNK_SIZE = 100
MIN_VALID_WORDS = 8

# Número máximo de tentativas de vision por imagem antes de
# desistir dela definitivamente (evita reprocessar para sempre
# uma imagem que o modelo nunca consegue descrever de forma válida).
MAX_VISION_ATTEMPTS = 3


# ============================================================
# ESCOPO DO DOCUMENTO
# ============================================================

def get_document_scope(pdf_path):
    """
    Determina se o documento é público ou interno.

    Estrutura esperada:

        documents/
            manual.pdf
            outro.pdf

            internal/
                documento-interno.pdf

    Qualquer PDF dentro de 'internal/' será considerado interno.
    """

    relative_path = pdf_path.relative_to(
        DOCUMENTS_DIR
    )

    if "internal" in relative_path.parts:
        return "internal"

    return "public"


# ============================================================
# CONFIGURAÇÃO VISION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

VISION_MODEL = "gemma3:4b"

VISION_NUM_PREDICT = 600

VISION_TIMEOUT = 600


# ============================================================
# HASH DO ARQUIVO
# ============================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# HASH DA IMAGEM
# ============================================================

def calculate_image_hash(image_bytes):

    return hashlib.sha256(
        image_bytes
    ).hexdigest()


# ============================================================
# IDENTIFICAÇÃO DO CHUNK DE TEXTO
# ============================================================

def generate_chunk_id(
    document,
    page,
    chunk,
    text
):

    content = (
        f"{document}|"
        f"{page}|"
        f"{chunk}|"
        f"{text}"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# IDENTIFICAÇÃO DO CHUNK VISION
# ============================================================

def generate_vision_chunk_id(
    document,
    page,
    image_index,
    image_hash
):

    content = (
        f"VISION|"
        f"{document}|"
        f"{page}|"
        f"{image_index}|"
        f"{image_hash}"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# LIMPEZA DO TEXTO
# ============================================================

def clean_text(text):

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = re.sub(
        r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}",
        "",
        text
    )

    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        "",
        text
    )

    lixo = [
        "Entrar BR",
        "Essas informações foram úteis?",
        "Precisa de ajuda?",
        "Pergunte ao Autodesk Assistant!",
        "Qual é o seu nível de suporte?",
        "Visualizar níveis de suporte",

        "Visão geral da empresa",
        "Carreiras",
        "Relação com investidores",
        "Autodesk Trust Center",
        "Autodesk Foundation",
        "Sustentabilidade",
        "Fale conosco",

        "Estudantes e educadores",
        "Como comprar",
        "Visualizar todos os produtos",
        "Comprar com a Autodesk",
        "Opções de renovação",
        "Localizar um parceiro",
        "Vendas e reembolsos",
        "Gerenciar sua conta",

        "Fazer o download e instalar o software",
        "Status do produto",
        "Comunidade Autodesk",
        "Notícias",
        "Inclusividade global",
        "LA28 Games",
        "Suporte educacional",
        "Entrar em contato com o suporte",

        "Privacidade",
        "Preferências de cookies",
        "Informar não conformidade",
        "Avisos legais",

        "Escolher seu plano de assinaturas",
        "Preço pré-pago para uso ocasional",

        "Pesquisa da Autodesk",
    ]

    for item in lixo:

        text = text.replace(
            item,
            ""
        )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# VALIDAÇÃO DO CHUNK DE TEXTO
# ============================================================

def is_valid_chunk(text):

    text = text.strip()

    if not text:
        return False

    if len(text) < MIN_VALID_CHUNK_SIZE:
        return False

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE
    )

    if len(words) < MIN_VALID_WORDS:
        return False

    lixo_patterns = [
        r"©\s*202\d",
        r"todos os direitos reservados",
        r"autodesk,?\s*\(?inglês\)?",
        r"\(inglês\)",
        r"escolher seu plano de assinaturas",
        r"preço pré-pago",
        r"entrar em contato com o suporte",
    ]

    lixo_score = 0

    for pattern in lixo_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        lixo_score += len(matches)

    if lixo_score >= 2:
        return False

    return True


# ============================================================
# VALIDAÇÃO DO TEXTO VISION
# ============================================================

def is_valid_vision_text(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 30:
        return False

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE
    )

    if len(words) < 5:
        return False

    return True


# ============================================================
# DIVISÃO EM CHUNKS
# ============================================================

def split_text(text):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        is_command = paragraph.startswith("/opt/")

        if is_command:

            if current:

                chunks.append(
                    current.strip()
                )

                current = ""

            chunks.append(
                paragraph
            )

            continue

        if (
            len(current)
            + len(paragraph)
            + 2
            <= CHUNK_SIZE
        ):

            current += (
                ("\n\n" if current else "")
                + paragraph
            )

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            current = paragraph

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# VISION - GEMMA
# ============================================================

def analyze_image_with_vision(
    image_bytes,
    page_number,
    image_index
):

    print(
        f"      Vision: página {page_number}, "
        f"imagem {image_index}"
    )

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = """
Analise esta imagem de um manual técnico para indexação em um sistema RAG.

Extraia somente informações úteis para responder perguntas futuras sobre o manual.

Inclua:
- textos visíveis importantes;
- nomes de softwares;
- nomes de menus, telas, botões e campos;
- configurações;
- valores exibidos;
- nomes de projetos;
- procedimentos ou passos mostrados;
- informações técnicas relevantes;
- relações entre elementos da interface.

Se a imagem mostrar uma sequência de passos, descreva a sequência.

Não invente informações que não estejam visíveis.

Escreva uma descrição objetiva e estruturada em português.
Não fale sobre a qualidade da imagem.
Não diga que você é uma IA.
Não use introduções desnecessárias.
"""

    payload = {

        "model": VISION_MODEL,

        "messages": [

            {
                "role": "user",

                "content": prompt,

                "images": [
                    image_base64
                ]
            }

        ],

        "stream": False,

        "options": {

            "num_predict": VISION_NUM_PREDICT

        }

    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=VISION_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        text = (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not text:

            print(
                "      Vision: resposta vazia"
            )

            return None

        print(
            f"      Vision: "
            f"{len(text)} caracteres"
        )

        return text

    except Exception as e:

        print(
            f"      ERRO Vision: {e}"
        )

        return None


# ============================================================
# PROCESSAR IMAGENS DO PDF
# ============================================================

def process_pdf_images(
    pdf_path,
    doc,
    file_hash,
    old_chunks,
    old_vision_failures=None
):
    """
    Retorna (results, vision_failures).

    vision_failures é um dict {image_hash: tentativas} usado para
    limitar quantas vezes uma mesma imagem é reenviada ao modelo
    de vision caso ela continue falhando na validação.
    """

    if old_vision_failures is None:
        old_vision_failures = {}

    results = []

    # Começa como cópia do estado anterior; vai sendo atualizado
    # conforme as imagens são (re)processadas nesta execução.
    vision_failures = dict(old_vision_failures)

    document_scope = get_document_scope(
        pdf_path
    )

    # --------------------------------------------------------
    # Vision antigo indexado pelo image_hash
    # --------------------------------------------------------

    old_vision_by_image_hash = {}

    for chunk in old_chunks:

        if chunk.get("source_type") != "vision":
            continue

        image_hash = chunk.get(
            "image_hash"
        )

        if image_hash:

            old_vision_by_image_hash[
                image_hash
            ] = chunk

    images_found = 0
    images_reused = 0
    images_processed = 0
    images_failed = 0
    images_skipped = 0

    # --------------------------------------------------------
    # Percorre páginas
    # --------------------------------------------------------

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        images = page.get_images(
            full=True
        )

        if not images:
            continue

        # ----------------------------------------------------
        # Percorre imagens
        # ----------------------------------------------------

        for image_index, image_info in enumerate(
            images,
            start=1
        ):

            xref = image_info[0]

            try:

                image_data = doc.extract_image(
                    xref
                )

                image_bytes = image_data[
                    "image"
                ]

            except Exception as e:

                print(
                    f"      Erro extraindo imagem "
                    f"página {page_number}, "
                    f"imagem {image_index}: {e}"
                )

                images_failed += 1

                continue

            images_found += 1

            image_hash = calculate_image_hash(
                image_bytes
            )

            # ------------------------------------------------
            # IMAGEM JÁ PROCESSADA COM SUCESSO
            # ------------------------------------------------

            if image_hash in old_vision_by_image_hash:

                old_chunk = old_vision_by_image_hash[
                    image_hash
                ]

                reused_chunk = dict(
                    old_chunk
                )

                reused_chunk[
                    "file_hash"
                ] = file_hash

                reused_chunk[
                    "document_scope"
                ] = document_scope

                results.append(
                    reused_chunk
                )

                images_reused += 1

                print(
                    f"      Imagem reutilizada: "
                    f"página {page_number}, "
                    f"imagem {image_index}"
                )

                continue

            # ------------------------------------------------
            # IMAGEM JÁ ESGOTOU TENTATIVAS DE VISION
            # ------------------------------------------------

            attempts_so_far = vision_failures.get(
                image_hash,
                0
            )

            if attempts_so_far >= MAX_VISION_ATTEMPTS:

                images_skipped += 1

                print(
                    f"      Vision: imagem ignorada "
                    f"(página {page_number}, "
                    f"imagem {image_index}) - "
                    f"já falhou {attempts_so_far}x"
                )

                continue

            # ------------------------------------------------
            # NOVA TENTATIVA DE VISION
            # ------------------------------------------------

            images_processed += 1

            vision_text = analyze_image_with_vision(
                image_bytes,
                page_number,
                image_index
            )

            if not is_valid_vision_text(
                vision_text
            ):

                print(
                    "      Vision: conteúdo "
                    "não considerado válido"
                )

                images_failed += 1

                vision_failures[
                    image_hash
                ] = attempts_so_far + 1

                continue

            # Sucesso: garante que a imagem não fique mais
            # marcada como "em falha".
            vision_failures.pop(
                image_hash,
                None
            )

            chunk_id = generate_vision_chunk_id(
                pdf_path.name,
                page_number,
                image_index,
                image_hash
            )

            results.append({

                "chunk_id": chunk_id,

                "document": pdf_path.name,

                "file_hash": file_hash,

                "page": page_number,

                "chunk": image_index,

                "source_type": "vision",

                "document_scope": document_scope,

                "image_index": image_index,

                "image_hash": image_hash,

                "text": vision_text

            })

    print()

    print(
        f"  Imagens encontradas  : {images_found}"
    )

    print(
        f"  Imagens reutilizadas  : {images_reused}"
    )

    print(
        f"  Imagens processadas   : {images_processed}"
    )

    print(
        f"  Imagens com erro      : {images_failed}"
    )

    print(
        f"  Imagens ignoradas (limite): {images_skipped}"
    )

    print(
        f"  Chunks vision         : {len(results)}"
    )

    return results, vision_failures


# ============================================================
# PROCESSAR PDF
# ============================================================

def process_pdf(
    pdf_path,
    file_hash,
    old_chunks=None,
    old_vision_failures=None
):
    """
    Retorna (results, vision_failures).
    """

    if old_chunks is None:

        old_chunks = []

    document_scope = get_document_scope(
        pdf_path
    )

    print()

    print(
        f"Processando: {pdf_path.name}"
    )

    print(
        f"Escopo: {document_scope}"
    )

    print(
        f"SHA256: {file_hash}"
    )

    doc = pymupdf.open(
        pdf_path
    )

    results = []

    paginas_vazias = 0

    chunks_descartados = 0

    # --------------------------------------------------------
    # PROCESSAMENTO DE TEXTO
    # --------------------------------------------------------

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        raw_text = page.get_text()

        if not raw_text.strip():

            paginas_vazias += 1

            print(
                f"  Página {page_number}: "
                f"sem texto"
            )

            continue

        text = clean_text(
            raw_text
        )

        if not text:

            paginas_vazias += 1

            print(
                f"  Página {page_number}: "
                f"vazia após limpeza"
            )

            continue

        chunks = split_text(
            text
        )

        valid_chunks = 0

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            if not is_valid_chunk(
                chunk
            ):

                chunks_descartados += 1

                continue

            chunk_id = generate_chunk_id(
                pdf_path.name,
                page_number,
                chunk_number,
                chunk
            )

            results.append({

                "chunk_id": chunk_id,

                "document": pdf_path.name,

                "file_hash": file_hash,

                "page": page_number,

                "chunk": chunk_number,

                "source_type": "text",

                "document_scope": document_scope,

                "text": chunk

            })

            valid_chunks += 1

        if valid_chunks == 0:

            print(
                f"  Página {page_number}: "
                f"nenhum chunk válido"
            )

    page_count = len(doc)

    # --------------------------------------------------------
    # PROCESSAMENTO VISION
    # --------------------------------------------------------

    print()

    print(
        "  --------------------------------------"
    )

    print(
        "  PROCESSAMENTO VISION"
    )

    print(
        "  --------------------------------------"
    )

    vision_chunks, vision_failures = process_pdf_images(
        pdf_path,
        doc,
        file_hash,
        old_chunks,
        old_vision_failures
    )

    results.extend(
        vision_chunks
    )

    doc.close()

    # --------------------------------------------------------
    # RESULTADO DO DOCUMENTO
    # --------------------------------------------------------

    print()

    print(
        f"  Páginas processadas: "
        f"{page_count - paginas_vazias}"
    )

    print(
        f"  Páginas sem conteúdo: "
        f"{paginas_vazias}"
    )

    print(
        f"  Chunks descartados: "
        f"{chunks_descartados}"
    )

    print(
        f"  Chunks gerados: "
        f"{len(results)}"
    )

    return results, vision_failures


# ============================================================
# CARREGAR JSON
# ============================================================

def load_json(
    file_path,
    default
):

    if not file_path.exists():

        return default

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"Erro lendo {file_path}: {e}"
        )

        return default


# ============================================================
# DOCUMENTO POSSUI VISION?
# ============================================================

def document_has_vision_chunks(
    chunks
):

    for chunk in chunks:

        if chunk.get(
            "source_type"
        ) == "vision":

            return True

    return False


# ============================================================
# VALIDAR NOMES DE DOCUMENTOS
# ============================================================

def validate_unique_document_names(
    pdf_files
):

    documents = {}

    for pdf in pdf_files:

        document_name = pdf.name

        documents.setdefault(
            document_name,
            []
        ).append(
            pdf
        )

    duplicates = {
        name: paths
        for name, paths in documents.items()
        if len(paths) > 1
    }

    if not duplicates:
        return True

    print()

    print(
        "ERRO: existem PDFs com o mesmo nome."
    )

    print(
        "O sistema utiliza o nome do arquivo "
        "como identidade do documento."
    )

    print()

    for document_name, paths in duplicates.items():

        print(
            f"Documento duplicado: {document_name}"
        )

        for path in paths:

            print(
                f"  - {path}"
            )

        print()

    print(
        "Renomeie os arquivos antes de executar "
        "a ingestão."
    )

    return False


# ============================================================
# ATUALIZAR CHUNKS REUTILIZADOS
# ============================================================

def update_reused_chunks(
    chunks,
    file_hash,
    document_scope
):

    updated_chunks = []

    for chunk in chunks:

        updated_chunk = dict(
            chunk
        )

        updated_chunk[
            "file_hash"
        ] = file_hash

        updated_chunk[
            "document_scope"
        ] = document_scope

        updated_chunks.append(
            updated_chunk
        )

    return updated_chunks


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Ingestão incremental de documentos PDF "
            "para o RAG."
        )
    )

    parser.add_argument(
        "--scope",
        choices=(
            "public",
            "internal",
            "all"
        ),
        default="public",
        help=(
            "Escopo dos documentos a processar. "
            "Padrão: public."
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "========================================"
    )

    print(
        "INGESTÃO INCREMENTAL DE DOCUMENTOS"
    )

    print(
        "TEXT + VISION"
    )

    print(
        "========================================"
    )

    print(
        f"Diretório: {DOCUMENTS_DIR}"
    )

    print(
        f"Vision model: {VISION_MODEL}"
    )

    print(
        f"Escopo selecionado: {args.scope}"
    )

    # --------------------------------------------------------
    # Localiza TODOS os PDFs
    # --------------------------------------------------------

    all_pdf_files = sorted(
        DOCUMENTS_DIR.rglob("*.pdf")
    )

    print(
        f"PDFs encontrados no diretório: "
        f"{len(all_pdf_files)}"
    )

    if not all_pdf_files:

        print(
            "\nNenhum PDF encontrado."
        )

        return

    # --------------------------------------------------------
    # Validação de nomes duplicados
    # --------------------------------------------------------

    if not validate_unique_document_names(
        all_pdf_files
    ):

        return

    # --------------------------------------------------------
    # Seleciona PDFs pelo escopo
    # --------------------------------------------------------

    selected_pdf_files = sorted(
        p
        for p in all_pdf_files
        if (
            args.scope == "all"
            or get_document_scope(p) == args.scope
        )
    )

    print(
        f"PDFs selecionados para ingestão "
        f"({args.scope}): "
        f"{len(selected_pdf_files)}"
    )

    if not selected_pdf_files:

        print(
            "\nNenhum PDF encontrado para "
            f"o escopo '{args.scope}'."
        )

        return

    # --------------------------------------------------------
    # Estado anterior
    # --------------------------------------------------------

    state = load_json(
        STATE_FILE,
        {}
    )

    # --------------------------------------------------------
    # Chunks anteriores
    # --------------------------------------------------------

    old_chunks = load_json(
        OUTPUT_FILE,
        []
    )

    # --------------------------------------------------------
    # Organiza chunks antigos por documento
    # --------------------------------------------------------

    old_chunks_by_document = {}

    for chunk in old_chunks:

        document = chunk.get(
            "document"
        )

        if not document:
            continue

        old_chunks_by_document.setdefault(
            document,
            []
        ).append(
            chunk
        )

    # --------------------------------------------------------
    # Documentos existentes fisicamente
    # --------------------------------------------------------

    all_current_documents = {
        pdf.name
        for pdf in all_pdf_files
    }

    selected_documents = {
        pdf.name
        for pdf in selected_pdf_files
    }

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    final_chunks = []

    new_documents = 0
    unchanged_documents = 0
    changed_documents = 0
    removed_documents = 0
    failed_documents = 0

    # --------------------------------------------------------
    # PRESERVA DOCUMENTOS FORA DO ESCOPO
    # --------------------------------------------------------
    #
    # Isto é fundamental.
    #
    # Se rodarmos:
    #
    #     --scope public
    #
    # os chunks internos já existentes permanecem.
    #
    # Se rodarmos:
    #
    #     --scope internal
    #
    # os chunks públicos já existentes permanecem.
    #
    # --------------------------------------------------------

    for document_name, document_chunks in (
        old_chunks_by_document.items()
    ):

        if document_name not in selected_documents:

            if document_name in all_current_documents:

                final_chunks.extend(
                    document_chunks
                )

    # --------------------------------------------------------
    # PROCESSA PDFs SELECIONADOS
    # --------------------------------------------------------

    for pdf in selected_pdf_files:

        document_name = pdf.name

        document_scope = get_document_scope(
            pdf
        )

        print()

        print(
            "----------------------------------------"
        )

        print(
            f"Documento: {document_name}"
        )

        print(
            f"Escopo: {document_scope}"
        )

        print(
            f"Caminho: {pdf}"
        )

        print(
            "Calculando SHA256..."
        )

        try:

            file_hash = calculate_file_hash(
                pdf
            )

        except Exception as e:

            print(
                f"  ERRO calculando hash de "
                f"{document_name}: {e}"
            )

            print(
                "  Documento será ignorado "
                "nesta execução."
            )

            failed_documents += 1

            # Se já existiam chunks desse documento,
            # preserva-os para não perder dados.

            if document_name in old_chunks_by_document:

                final_chunks.extend(
                    old_chunks_by_document[
                        document_name
                    ]
                )

            continue

        previous_state = state.get(
            document_name,
            {}
        )

        previous_hash = previous_state.get(
            "sha256"
        )

        previous_vision_failures = previous_state.get(
            "vision_failures",
            {}
        )

        document_old_chunks = (
            old_chunks_by_document.get(
                document_name,
                []
            )
        )

        has_vision = document_has_vision_chunks(
            document_old_chunks
        )

        # ----------------------------------------------------
        # PDF NÃO MUDOU E JÁ POSSUI VISION
        # ----------------------------------------------------

        if (
            previous_hash == file_hash
            and document_name in old_chunks_by_document
            and has_vision
        ):

            print(
                "STATUS: INALTERADO"
            )

            print(
                "Chunks antigos serão reutilizados."
            )

            reused_chunks = update_reused_chunks(
                document_old_chunks,
                file_hash,
                document_scope
            )

            final_chunks.extend(
                reused_chunks
            )

            unchanged_documents += 1

            state[
                document_name
            ] = {

                "sha256": file_hash,

                "chunks": len(
                    reused_chunks
                ),

                "document_scope": document_scope,

                "vision_failures": previous_vision_failures

            }

            continue

        # ----------------------------------------------------
        # PDF NÃO MUDOU MAS AINDA NÃO POSSUI VISION
        # ----------------------------------------------------

        if (
            previous_hash == file_hash
            and document_name in old_chunks_by_document
            and not has_vision
        ):

            print(
                "STATUS: INALTERADO"
            )

            print(
                "Texto será reutilizado."
            )

            print(
                "Vision ainda não processado "
                "(ou pendente de novas tentativas)."
            )

            text_chunks = []

            for chunk in document_old_chunks:

                if chunk.get(
                    "source_type",
                    "text"
                ) == "vision":

                    continue

                reused_chunk = dict(
                    chunk
                )

                reused_chunk[
                    "file_hash"
                ] = file_hash

                reused_chunk[
                    "document_scope"
                ] = document_scope

                text_chunks.append(
                    reused_chunk
                )

            try:

                chunks, vision_failures = process_pdf(
                    pdf,
                    file_hash,
                    text_chunks,
                    previous_vision_failures
                )

            except Exception as e:

                print(
                    f"  ERRO processando "
                    f"{document_name}: {e}"
                )

                print(
                    "  Chunks anteriores (sem vision) "
                    "serão mantidos; documento será "
                    "tentado novamente na próxima execução."
                )

                failed_documents += 1

                final_chunks.extend(
                    document_old_chunks
                )

                continue

            final_chunks.extend(
                chunks
            )

            state[
                document_name
            ] = {

                "sha256": file_hash,

                "chunks": len(chunks),

                "document_scope": document_scope,

                "vision_failures": vision_failures

            }

            continue

        # ----------------------------------------------------
        # PDF NOVO
        # ----------------------------------------------------

        if previous_hash is None:

            print(
                "STATUS: NOVO DOCUMENTO"
            )

            new_documents += 1

        # ----------------------------------------------------
        # PDF ALTERADO
        # ----------------------------------------------------

        else:

            print(
                "STATUS: DOCUMENTO ALTERADO"
            )

            changed_documents += 1

        # ----------------------------------------------------
        # Processamento completo
        # ----------------------------------------------------

        try:

            chunks, vision_failures = process_pdf(
                pdf,
                file_hash,
                document_old_chunks,
                previous_vision_failures
            )

        except Exception as e:

            print(
                f"  ERRO processando {document_name}: {e}"
            )

            failed_documents += 1

            if document_name in old_chunks_by_document:

                print(
                    "  Chunks da versão anterior serão "
                    "mantidos; documento será tentado "
                    "novamente na próxima execução."
                )

                final_chunks.extend(
                    document_old_chunks
                )

            else:

                print(
                    "  Documento não possui chunks "
                    "anteriores; será ignorado nesta "
                    "execução e tentado novamente na "
                    "próxima."
                )

            continue

        final_chunks.extend(
            chunks
        )

        # ----------------------------------------------------
        # Atualiza estado
        # ----------------------------------------------------

        state[
            document_name
        ] = {

            "sha256": file_hash,

            "chunks": len(chunks),

            "document_scope": document_scope,

            "vision_failures": vision_failures

        }

    # --------------------------------------------------------
    # Detecta PDFs removidos
    # --------------------------------------------------------
    #
    # IMPORTANTE:
    #
    # A comparação é feita contra TODOS os PDFs físicos,
    # e não somente contra o escopo selecionado.
    #
    # Isso evita que --scope public interprete os documentos
    # internos como removidos.
    #
    # --------------------------------------------------------

    previous_documents = set(
        state.keys()
    )

    removed = (
        previous_documents
        - all_current_documents
    )

    for document_name in removed:

        print()

        print(
            f"Documento removido: "
            f"{document_name}"
        )

        removed_documents += 1

    # --------------------------------------------------------
    # Remove documentos realmente apagados
    # --------------------------------------------------------

    for document_name in removed:

        state.pop(
            document_name,
            None
        )

    # --------------------------------------------------------
    # Salva chunks
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Salva estado
    # --------------------------------------------------------

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    text_chunks_count = sum(
        1
        for chunk in final_chunks
        if chunk.get(
            "source_type",
            "text"
        ) == "text"
    )

    vision_chunks_count = sum(
        1
        for chunk in final_chunks
        if chunk.get(
            "source_type"
        ) == "vision"
    )

    public_chunks_count = sum(
        1
        for chunk in final_chunks
        if chunk.get(
            "document_scope"
        ) == "public"
    )

    internal_chunks_count = sum(
        1
        for chunk in final_chunks
        if chunk.get(
            "document_scope"
        ) == "internal"
    )

    public_documents = sum(
        1
        for pdf in all_pdf_files
        if get_document_scope(pdf) == "public"
    )

    internal_documents = sum(
        1
        for pdf in all_pdf_files
        if get_document_scope(pdf) == "internal"
    )

    # Calculado a partir do resultado final (não de um contador
    # incrementado durante o loop), para refletir de fato quantos
    # documentos distintos possuem pelo menos um chunk de vision.

    documents_with_vision = len({
        chunk.get("document")
        for chunk in final_chunks
        if chunk.get("source_type") == "vision"
    })

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print()

    print(
        "========================================"
    )

    print(
        "INGESTÃO CONCLUÍDA"
    )

    print(
        "========================================"
    )

    print(
        f"Escopo processado  : {args.scope}"
    )

    print(
        f"PDFs no diretório  : {len(all_pdf_files)}"
    )

    print(
        f"PDFs selecionados  : {len(selected_pdf_files)}"
    )

    print(
        f"PDFs públicos      : {public_documents}"
    )

    print(
        f"PDFs internos      : {internal_documents}"
    )

    print(
        f"PDFs novos         : {new_documents}"
    )

    print(
        f"PDFs inalterados   : {unchanged_documents}"
    )

    print(
        f"PDFs alterados     : {changed_documents}"
    )

    print(
        f"PDFs removidos     : {removed_documents}"
    )

    print(
        f"PDFs com erro      : {failed_documents}"
    )

    print(
        f"Docs com Vision    : {documents_with_vision}"
    )

    print(
        f"Chunks de texto    : {text_chunks_count}"
    )

    print(
        f"Chunks Vision      : {vision_chunks_count}"
    )

    print(
        f"Chunks públicos    : {public_chunks_count}"
    )

    print(
        f"Chunks internos    : {internal_chunks_count}"
    )

    print(
        f"Total de chunks    : {len(final_chunks)}"
    )

    print(
        f"Chunks antigos     : {len(old_chunks)}"
    )

    print(
        f"Arquivo chunks     : {OUTPUT_FILE}"
    )

    print(
        f"Arquivo estado     : {STATE_FILE}"
    )

    print(
        "========================================"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()