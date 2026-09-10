import json
from pydoc import doc
import re
from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DOCUMENTS_DIR = Path("/opt/ai/rag/documents")
OUTPUT_FILE = Path("/opt/ai/rag/database/chunks.json")

CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 300


# ============================================================
# LIMPEZA DO TEXTO
# ============================================================

def clean_text(text):

    text = text.replace("\xa0", " ")

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remove timestamps
    text = re.sub(
        r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}",
        "",
        text
    )

    # Remove paginação
    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        "",
        text
    )

    # Elementos de interface / rodapé
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
        "Programa de afiliados",
        "Pesquisa da Autodesk",
        "Escolher seu plano de assinaturas",
        "Preço pré-pago para uso ocasional",
        "Suporte educacional",
        "Entrar em contato com o suporte",
        "Privacidade",
        "Preferências de cookies",
        "Informar não conformidade",
        "Avisos legais",
    ]

    for item in lixo:
        text = text.replace(item, "")

    # Normaliza espaços
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Mantém quebras de linha
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# DIVISÃO EM CHUNKS
# ============================================================

def split_text(text):
    """
    Divide o texto por parágrafos.

    Tenta manter comandos completos.
    """

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

        # Detecta comandos
        is_command = paragraph.startswith("/opt/")

        # Comandos ficam isolados
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

        # Parágrafo cabe no chunk atual
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

    # Último chunk
    if current:
        chunks.append(
            current.strip()
        )

    # Junta chunks muito pequenos
    final_chunks = []

    for chunk in chunks:

        if (
            final_chunks
            and len(chunk) < MIN_CHUNK_SIZE
            and not chunk.startswith("/opt/")
        ):

            final_chunks[-1] += (
                "\n\n" + chunk
            )

        else:

            final_chunks.append(
                chunk
            )

    return final_chunks


# ============================================================
# PROCESSAR PDF
# ============================================================

def process_pdf(pdf_path):

    print(
        f"\nProcessando: {pdf_path.name}"
    )

    doc = pymupdf.open(pdf_path)

    results = []

    # Processa TODAS as páginas
    for page_number, page in enumerate(
        doc,
        start=1
    ):

        raw_text = page.get_text()

        if not raw_text.strip():
            continue

        text = clean_text(
            raw_text
        )

        if not text:
            continue

        chunks = split_text(
            text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            results.append({
                "document": pdf_path.name,
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk
            })

    page_count = len(doc)

    doc.close()

    print(
    f"  Páginas processadas: {page_count}"
    )
 
    print(
    f"  Chunks gerados: {len(results)}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    all_chunks = []

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    print(
        "========================================"
    )

    print(
        "INGESTÃO DE DOCUMENTOS"
    )

    print(
        "========================================"
    )

    print(
        f"Diretório: {DOCUMENTS_DIR}"
    )

    print(
        f"PDFs encontrados: {len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nNenhum PDF encontrado."
        )

        return

    # Processa todos os PDFs
    for pdf in pdf_files:

        chunks = process_pdf(
            pdf
        )

        all_chunks.extend(
            chunks
        )

    # Cria database se necessário
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva chunks
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
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
        f"PDFs processados: {len(pdf_files)}"
    )

    print(
        f"Total de chunks: {len(all_chunks)}"
    )

    print(
        f"Arquivo: {OUTPUT_FILE}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()