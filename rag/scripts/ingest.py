import json
import re
import hashlib
import base64
import requests

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
# IDENTIFICAÇÃO DO CHUNK
# ============================================================

def generate_chunk_id(document, page, chunk, text):

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
# VALIDAÇÃO
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
# VALIDAÇÃO VISION
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
    document_name,
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
    old_chunks
):

    results = []

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

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        images = page.get_images(
            full=True
        )

        if not images:
            continue

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
            # IMAGEM JÁ PROCESSADA
            # ------------------------------------------------

            if image_hash in old_vision_by_image_hash:

                old_chunk = old_vision_by_image_hash[
                    image_hash
                ]

                results.append(
                    old_chunk
                )

                images_reused += 1

                print(
                    f"      Imagem reutilizada: "
                    f"página {page_number}, "
                    f"imagem {image_index}"
                )

                continue

            # ------------------------------------------------
            # NOVA IMAGEM
            # ------------------------------------------------

            images_processed += 1

            vision_text = analyze_image_with_vision(
                image_bytes,
                pdf_path.name,
                page_number,
                image_index
            )

            if not is_valid_vision_text(
                vision_text
            ):

                print(
                    f"      Vision: conteúdo "
                    f"não considerado válido"
                )

                images_failed += 1

                continue

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

                "image_index": image_index,

                "image_hash": image_hash,

                "text": vision_text

            })

    print()

    print(
        f"  Imagens encontradas : {images_found}"
    )

    print(
        f"  Imagens reutilizadas : {images_reused}"
    )

    print(
        f"  Imagens processadas  : {images_processed}"
    )

    print(
        f"  Imagens com erro     : {images_failed}"
    )

    print(
        f"  Chunks vision        : {len(results)}"
    )

    return results


# ============================================================
# PROCESSAR PDF
# ============================================================

def process_pdf(
    pdf_path,
    file_hash,
    old_chunks=None
):

    if old_chunks is None:

        old_chunks = []

    print()

    print(
        f"Processando: {pdf_path.name}"
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

                "text": chunk

            })

            valid_chunks += 1

        if valid_chunks == 0:

            print(
                f"  Página {page_number}: "
                f"nenhum chunk válido"
            )

    page_count = len(doc)

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

    vision_chunks = process_pdf_images(
        pdf_path,
        doc,
        file_hash,
        old_chunks
    )

    results.extend(
        vision_chunks
    )

    doc.close()

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

    return results


# ============================================================
# CARREGAR JSON
# ============================================================

def load_json(file_path, default):

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
# MAIN
# ============================================================

def main():

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

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    print(
        f"PDFs encontrados: {len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nNenhum PDF encontrado."
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

        document = chunk["document"]

        old_chunks_by_document.setdefault(
            document,
            []
        ).append(
            chunk
        )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    final_chunks = []

    new_documents = 0

    unchanged_documents = 0

    changed_documents = 0

    removed_documents = 0

    vision_documents = 0

    # --------------------------------------------------------
    # Processa PDFs
    # --------------------------------------------------------

    current_documents = set()

    for pdf in pdf_files:

        document_name = pdf.name

        current_documents.add(
            document_name
        )

        print()

        print(
            "----------------------------------------"
        )

        print(
            f"Documento: {document_name}"
        )

        print(
            "Calculando SHA256..."
        )

        file_hash = calculate_file_hash(
            pdf
        )

        previous_hash = (
            state
            .get(document_name, {})
            .get("sha256")
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
        # PDF não mudou E já possui vision
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

            final_chunks.extend(
                document_old_chunks
            )

            unchanged_documents += 1

            continue

        # ----------------------------------------------------
        # PDF não mudou MAS ainda não possui vision
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
                "Vision ainda não processado."
            )

            text_chunks = [
                chunk
                for chunk in document_old_chunks
                if chunk.get(
                    "source_type",
                    "text"
                ) != "vision"
            ]

            chunks = process_pdf(
                pdf,
                file_hash,
                text_chunks
            )

            final_chunks.extend(
                chunks
            )

            vision_documents += 1

            continue

        # ----------------------------------------------------
        # PDF novo
        # ----------------------------------------------------

        if previous_hash is None:

            print(
                "STATUS: NOVO DOCUMENTO"
            )

            new_documents += 1

        # ----------------------------------------------------
        # PDF alterado
        # ----------------------------------------------------

        else:

            print(
                "STATUS: DOCUMENTO ALTERADO"
            )

            changed_documents += 1

        chunks = process_pdf(
            pdf,
            file_hash,
            document_old_chunks
        )

        final_chunks.extend(
            chunks
        )

        vision_documents += 1

        # ----------------------------------------------------
        # Atualiza estado
        # ----------------------------------------------------

        state[document_name] = {

            "sha256": file_hash,

            "chunks": len(chunks)

        }

    # --------------------------------------------------------
    # Detecta PDFs removidos
    # --------------------------------------------------------

    previous_documents = set(
        state.keys()
    )

    removed = (
        previous_documents
        - current_documents
    )

    for document_name in removed:

        print()

        print(
            f"Documento removido: "
            f"{document_name}"
        )

        removed_documents += 1

    # --------------------------------------------------------
    # Remove documentos antigos do estado
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
    # Resultado
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
        f"PDFs novos       : {new_documents}"
    )

    print(
        f"PDFs inalterados : {unchanged_documents}"
    )

    print(
        f"PDFs alterados   : {changed_documents}"
    )

    print(
        f"PDFs removidos   : {removed_documents}"
    )

    print(
        f"Docs com Vision  : {vision_documents}"
    )

    print(
        f"Chunks de texto  : {text_chunks_count}"
    )

    print(
        f"Chunks Vision    : {vision_chunks_count}"
    )

    print(
        f"Total de chunks  : {len(final_chunks)}"
    )

    print(
        f"Chunks antigos   : "
        f"{len(old_chunks)}"
    )

    print(
        f"Arquivo chunks   : {OUTPUT_FILE}"
    )

    print(
        f"Arquivo estado   : {STATE_FILE}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()