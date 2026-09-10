import json
import re
from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DOCUMENTS_DIR = Path("/opt/ai/rag/documents")
OUTPUT_FILE = Path("/opt/ai/rag/database/chunks.json")

CHUNK_SIZE = 1500

# IMPORTANTE:
# Este valor NÃO é usado para descartar páginas.
# Serve apenas para ajudar a identificar chunks muito pequenos.
MIN_VALID_CHUNK_SIZE = 100
MIN_VALID_WORDS = 8


# ============================================================
# LIMPEZA DO TEXTO
# ============================================================

def clean_text(text):

    # Espaço não separável
    text = text.replace("\xa0", " ")

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove timestamps
    # --------------------------------------------------------

    text = re.sub(
        r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove paginação do tipo 12 / 300
    # --------------------------------------------------------

    text = re.sub(
        r"\b\d+\s*/\s*\d+\b",
        "",
        text
    )

    # --------------------------------------------------------
    # Elementos de interface / rodapé
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normaliza espaços
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normaliza quebras de linha
    # --------------------------------------------------------

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# VALIDAÇÃO DE CHUNK
# ============================================================

def is_valid_chunk(text):

    text = text.strip()

    if not text:
        return False

    # --------------------------------------------------------
    # Chunks extremamente pequenos
    # --------------------------------------------------------

    if len(text) < MIN_VALID_CHUNK_SIZE:
        return False

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE
    )

    if len(words) < MIN_VALID_WORDS:
        return False

    # --------------------------------------------------------
    # Detecta páginas/chunks claramente inúteis
    # --------------------------------------------------------

    linhas = [
        linha.strip()
        for linha in text.splitlines()
        if linha.strip()
    ]

    # Muito pouco conteúdo real
    if len(linhas) == 0:
        return False

    # --------------------------------------------------------
    # Padrões de páginas de capa / créditos
    # --------------------------------------------------------

    texto_lower = text.lower()

    padroes_lixo = [

        "todos os direitos reservados",

        "escolher seu plano de assinaturas",

        "preço pré-pago para uso ocasional",

        "entrar em contato com o suporte",

    ]

    ocorrencias_lixo = sum(
        1
        for padrao in padroes_lixo
        if padrao in texto_lower
    )

    # Se houver muitos elementos de rodapé,
    # provavelmente não é conteúdo útil.
    if ocorrencias_lixo >= 2:
        return False

    return True


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

        # ----------------------------------------------------
        # Detecta comandos
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Parágrafo cabe no chunk atual
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Último chunk
    # --------------------------------------------------------

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# PROCESSAR PDF
# ============================================================

def process_pdf(pdf_path):

    print()
    print(
        f"Processando: {pdf_path.name}"
    )

    doc = pymupdf.open(
        pdf_path
    )

    results = []

    paginas_processadas = 0
    paginas_vazias = 0
    chunks_descartados = 0

    # --------------------------------------------------------
    # Processa TODAS as páginas
    # --------------------------------------------------------

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        raw_text = page.get_text()

        # ----------------------------------------------------
        # Página sem texto
        # ----------------------------------------------------

        if not raw_text.strip():

            paginas_vazias += 1

            print(
                f"  Página {page_number}: "
                f"sem texto"
            )

            continue

        # ----------------------------------------------------
        # Limpeza
        # ----------------------------------------------------

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

        paginas_processadas += 1

        # ----------------------------------------------------
        # Divide em chunks
        # ----------------------------------------------------

        chunks = split_text(
            text
        )

        valid_chunks = 0

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            # ------------------------------------------------
            # Validação ocorre AQUI
            #
            # NÃO descartamos mais uma página inteira
            # por possuir pouco texto.
            # ------------------------------------------------

            if not is_valid_chunk(
                chunk
            ):

                chunks_descartados += 1

                print(
                    f"  Página {page_number}, "
                    f"chunk {chunk_number}: "
                    f"descartado por baixa qualidade"
                )

                continue

            results.append({

                "document": pdf_path.name,

                "page": page_number,

                "chunk": chunk_number,

                "text": chunk

            })

            valid_chunks += 1

        # ----------------------------------------------------
        # Página sem nenhum chunk válido
        # ----------------------------------------------------

        if valid_chunks == 0:

            print(
                f"  Página {page_number}: "
                f"nenhum chunk válido"
            )

    page_count = len(doc)

    doc.close()

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    paginas_descartadas = (
        paginas_vazias
    )

    print(
        f"  Páginas processadas: "
        f"{paginas_processadas}"
    )

    print(
        f"  Páginas sem conteúdo: "
        f"{paginas_descartadas}"
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
# MAIN
# ============================================================

def main():

    all_chunks = []

    # --------------------------------------------------------
    # Localiza PDFs
    # --------------------------------------------------------

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
        f"PDFs encontrados: "
        f"{len(pdf_files)}"
    )

    if not pdf_files:

        print(
            "\nNenhum PDF encontrado."
        )

        return

    # --------------------------------------------------------
    # Processa todos os PDFs
    # --------------------------------------------------------

    for pdf in pdf_files:

        chunks = process_pdf(
            pdf
        )

        all_chunks.extend(
            chunks
        )

    # --------------------------------------------------------
    # Cria diretório do banco
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
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
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Estatísticas finais
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
        f"PDFs processados: "
        f"{len(pdf_files)}"
    )

    print(
        f"Total de chunks: "
        f"{len(all_chunks)}"
    )

    print(
        f"Arquivo: "
        f"{OUTPUT_FILE}"
    )

    print(
        "========================================"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()