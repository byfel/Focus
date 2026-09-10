import json
import requests
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

INPUT_FILE = Path(
    "/opt/ai/rag/database/chunks.json"
)

OUTPUT_FILE = Path(
    "/opt/ai/rag/database/embeddings.json"
)

OLLAMA_URL = (
    "http://localhost:11434/api/embed"
)

MODEL = "embeddinggemma:latest"


# ============================================================
# CHAVE DE COMPATIBILIDADE
# ============================================================

def get_legacy_key(item):
    """
    Chave usada pelo embed.py antigo.

    Serve somente para localizar embeddings
    antigos que ainda não possuem chunk_id.
    """

    return (
        f"{item.get('document')}"
        f"|{item.get('page')}"
        f"|{item.get('chunk')}"
    )


# ============================================================
# CARREGAR JSON
# ============================================================

def load_json(path, default):

    if not path.exists():
        return default

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"ERRO lendo {path}: {e}"
        )

        raise


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GERAÇÃO INCREMENTAL DE EMBEDDINGS")
    print("=" * 80)

    print()
    print(f"Modelo: {MODEL}")
    print(f"Chunks: {INPUT_FILE}")
    print(f"Embeddings: {OUTPUT_FILE}")


    # --------------------------------------------------------
    # 1. Carrega chunks
    # --------------------------------------------------------

    chunks = load_json(
        INPUT_FILE,
        []
    )

    print()
    print(
        f"Chunks encontrados: {len(chunks)}"
    )

    if not chunks:

        print(
            "Nenhum chunk encontrado."
        )

        return


    # --------------------------------------------------------
    # 2. Valida chunk_id
    # --------------------------------------------------------

    chunks_by_id = {}

    chunks_by_legacy_key = {}

    for chunk in chunks:

        chunk_id = chunk.get(
            "chunk_id"
        )

        if not chunk_id:

            raise ValueError(
                "Encontrado chunk sem chunk_id "
                "no chunks.json.\n"
                f"Documento: {chunk.get('document')}\n"
                f"Página: {chunk.get('page')}\n"
                f"Chunk: {chunk.get('chunk')}"
            )

        chunks_by_id[chunk_id] = chunk

        legacy_key = get_legacy_key(
            chunk
        )

        chunks_by_legacy_key[
            legacy_key
        ] = chunk


    print(
        f"Chunks com chunk_id: "
        f"{len(chunks_by_id)}"
    )


    # --------------------------------------------------------
    # 3. Carrega embeddings existentes
    # --------------------------------------------------------

    old_embeddings = load_json(
        OUTPUT_FILE,
        []
    )

    print(
        f"Embeddings existentes: "
        f"{len(old_embeddings)}"
    )


    # --------------------------------------------------------
    # 4. Indexa embeddings antigos
    # --------------------------------------------------------

    embeddings_by_chunk_id = {}

    embeddings_by_legacy_key = {}

    antigos_com_id = 0

    antigos_sem_id = 0


    for item in old_embeddings:

        chunk_id = item.get(
            "chunk_id"
        )

        if chunk_id:

            embeddings_by_chunk_id[
                chunk_id
            ] = item

            antigos_com_id += 1

        else:

            legacy_key = get_legacy_key(
                item
            )

            embeddings_by_legacy_key[
                legacy_key
            ] = item

            antigos_sem_id += 1


    print()
    print(
        f"Embeddings com chunk_id: "
        f"{antigos_com_id}"
    )

    print(
        f"Embeddings sem chunk_id: "
        f"{antigos_sem_id}"
    )


    # --------------------------------------------------------
    # 5. Processa chunks
    # --------------------------------------------------------

    results = []

    reutilizados = 0

    reconstruidos = 0

    novos = 0

    total = len(chunks)


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk_id = chunk[
            "chunk_id"
        ]

        legacy_key = get_legacy_key(
            chunk
        )


        # ====================================================
        # CASO 1
        # Embedding já possui chunk_id
        # ====================================================

        if chunk_id in embeddings_by_chunk_id:

            old = embeddings_by_chunk_id[
                chunk_id
            ]

            item = {
                **chunk,
                "embedding": old[
                    "embedding"
                ]
            }

            results.append(
                item
            )

            reutilizados += 1

            print(
                f"[{index}/{total}] "
                f"REUTILIZANDO ID "
                f"{chunk_id[:12]}... "
                f"{chunk['document']} "
                f"Página {chunk['page']} "
                f"Chunk {chunk['chunk']}"
            )

            continue


        # ====================================================
        # CASO 2
        # Embedding antigo sem chunk_id
        #
        # Podemos reutilizar porque a chave
        # documento + página + chunk coincide.
        # ====================================================

        if legacy_key in embeddings_by_legacy_key:

            old = embeddings_by_legacy_key[
                legacy_key
            ]

            item = {
                **chunk,
                "embedding": old[
                    "embedding"
                ]
            }

            results.append(
                item
            )

            reconstruidos += 1

            print(
                f"[{index}/{total}] "
                f"RECONSTRUINDO chunk_id "
                f"{chunk['document']} "
                f"Página {chunk['page']} "
                f"Chunk {chunk['chunk']}"
            )

            continue


        # ====================================================
        # CASO 3
        # Novo embedding
        # ====================================================

        text = chunk[
            "text"
        ]

        print(
            f"[{index}/{total}] "
            f"NOVO EMBEDDING → "
            f"{chunk['document']} "
            f"Página {chunk['page']} "
            f"Chunk {chunk['chunk']} "
            f"({len(text)} caracteres)"
        )


        response = requests.post(

            OLLAMA_URL,

            json={
                "model": MODEL,
                "input": text
            },

            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        embedding = data[
            "embeddings"
        ][0]


        item = {

            **chunk,

            "embedding": embedding

        }


        results.append(
            item
        )

        novos += 1


        print(
            f"    Embedding: "
            f"{len(embedding)} dimensões"
        )


    # --------------------------------------------------------
    # 6. Validação final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDAÇÃO")
    print("=" * 80)

    ids = [
        item.get("chunk_id")
        for item in results
    ]

    ids_validos = [
        x for x in ids
        if x
    ]

    duplicados = (
        len(ids_validos)
        - len(set(ids_validos))
    )


    print(
        f"Chunks de entrada       : "
        f"{len(chunks)}"
    )

    print(
        f"Embeddings de saída     : "
        f"{len(results)}"
    )

    print(
        f"Com chunk_id            : "
        f"{len(ids_validos)}"
    )

    print(
        f"IDs duplicados          : "
        f"{duplicados}"
    )


    if len(results) != len(chunks):

        raise ValueError(
            "Quantidade de embeddings "
            "diferente da quantidade de chunks."
        )


    if len(ids_validos) != len(results):

        raise ValueError(
            "Existe embedding sem chunk_id."
        )


    if duplicados != 0:

        raise ValueError(
            "Existem chunk_ids duplicados."
        )


    # --------------------------------------------------------
    # 7. Salva embeddings
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # 8. Resultado
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EMBEDDING CONCLUÍDO")
    print("=" * 80)

    print(
        f"Chunks encontrados : "
        f"{total}"
    )

    print(
        f"Reutilizados       : "
        f"{reutilizados}"
    )

    print(
        f"IDs reconstruídos   : "
        f"{reconstruidos}"
    )

    print(
        f"Novos embeddings    : "
        f"{novos}"
    )

    print(
        f"Total final         : "
        f"{len(results)}"
    )

    print(
        f"Arquivo             : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 80)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()