import json
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# ============================================================
# CONFIGURAÇÃO
# ============================================================

EMBEDDINGS_FILE = (
    "/opt/ai/rag/database/embeddings.json"
)

QDRANT_URL = (
    "http://localhost:6333"
)

COLLECTION_NAME = (
    "rag_documents"
)

# Tamanho máximo de cada lote enviado ao Qdrant.
#
# 500 pontos é suficientemente pequeno para evitar
# ultrapassar o limite de 32 MB do Qdrant.
BATCH_SIZE = 500


# ============================================================
# ID DETERMINÍSTICO
# ============================================================

def generate_qdrant_id(chunk_id):

    """
    Converte o chunk_id em um ID inteiro estável.

    O mesmo chunk_id SEMPRE produzirá o mesmo
    ID no Qdrant.
    """

    value = int(
        hashlib.sha256(
            chunk_id.encode("utf-8")
        ).hexdigest()[:16],
        16
    )

    return value & 0x7FFFFFFFFFFFFFFF


# ============================================================
# CARREGA IDS DO QDRANT
# ============================================================

def get_qdrant_ids(client):

    ids = set()

    offset = None

    while True:

        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        for record in records:

            ids.add(record.id)

        if offset is None:
            break

    return ids


# ============================================================
# UPSERT EM LOTES
# ============================================================

def upsert_in_batches(
    client,
    points,
    batch_size=BATCH_SIZE
):

    """
    Envia os pontos ao Qdrant em lotes.

    Evita enviar um payload gigantesco em uma única
    requisição HTTP e mantém o consumo de memória
    e o tamanho das requisições sob controle.
    """

    total = len(points)

    if total == 0:

        print(
            "Nenhum ponto novo para inserir."
        )

        return

    print()
    print(
        f"Enviando {total} pontos "
        f"em lotes de {batch_size}..."
    )

    for i in range(
        0,
        total,
        batch_size
    ):

        batch = points[
            i:i + batch_size
        ]

        batch_number = (
            (i // batch_size) + 1
        )

        total_batches = (
            (total + batch_size - 1)
            // batch_size
        )

        print(
            f"  Lote {batch_number}/"
            f"{total_batches}: "
            f"{i + 1}-{i + len(batch)} "
            f"/ {total}"
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

    print()
    print(
        "Todos os lotes foram "
        "sincronizados com sucesso."
    )


# ============================================================
# DELETE EM LOTES
# ============================================================

def delete_in_batches(
    client,
    ids,
    batch_size=BATCH_SIZE
):

    """
    Remove IDs obsoletos do Qdrant em lotes.
    """

    total = len(ids)

    if total == 0:

        print(
            "Nenhum ponto obsoleto para remover."
        )

        return

    print()
    print(
        f"Removendo {total} pontos "
        f"em lotes de {batch_size}..."
    )

    for i in range(
        0,
        total,
        batch_size
    ):

        batch = ids[
            i:i + batch_size
        ]

        batch_number = (
            (i // batch_size) + 1
        )

        total_batches = (
            (total + batch_size - 1)
            // batch_size
        )

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=batch,
            wait=True,
        )

        print(
            f"  Lote {batch_number}/"
            f"{total_batches}: "
            f"{i + len(batch)} / {total} "
            f"removidos"
        )

    print(
        "Pontos obsoletos removidos."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MIGRAÇÃO INCREMENTAL PARA QDRANT")
    print("=" * 80)

    # --------------------------------------------------------
    # Carrega embeddings
    # --------------------------------------------------------

    print()
    print("Carregando embeddings...")

    with open(
        EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        f"Embeddings encontrados: {len(data)}"
    )

    # --------------------------------------------------------
    # Conecta ao Qdrant
    # --------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print(
        f"Points atualmente no Qdrant: "
        f"{collection_info.points_count}"
    )

    # --------------------------------------------------------
    # Lê IDs atuais do Qdrant
    # --------------------------------------------------------

    print()
    print(
        "Lendo IDs existentes no Qdrant..."
    )

    existing_ids = get_qdrant_ids(
        client
    )

    print(
        f"IDs existentes: {len(existing_ids)}"
    )

    # --------------------------------------------------------
    # Prepara pontos
    # --------------------------------------------------------

    novos_points = []

    expected_ids = set()

    documentos = set()

    for item in data:

        chunk_id = item.get(
            "chunk_id"
        )

        if not chunk_id:

            raise ValueError(
                "Encontrado embedding sem "
                "chunk_id.\n"
                f"Documento: {item.get('document')}\n"
                f"Página: {item.get('page')}\n"
                f"Chunk: {item.get('chunk')}\n\n"
                "O embeddings.json precisa ser "
                "gerado pelo novo embed.py."
            )

        qdrant_id = generate_qdrant_id(
            chunk_id
        )

        expected_ids.add(
            qdrant_id
        )

        documentos.add(
            item["document"]
        )

        # ----------------------------------------------------
        # Só prepara para envio se ainda não existir
        # ----------------------------------------------------

        if qdrant_id not in existing_ids:

            novos_points.append(
                PointStruct(

                    id=qdrant_id,

                    vector=item["embedding"],

                    payload={

                        "chunk_id": chunk_id,

                        "document": item["document"],

                        "file_hash": item.get(
                            "file_hash"
                        ),

                        "page": item["page"],

                        "chunk": item["chunk"],

                        "source_type": item.get(
                            "source_type",
                            "text"
                        ),

                        "image_index": item.get(
                            "image_index"
                        ),

                        "image_hash": item.get(
                            "image_hash"
                        ),

                        "text": item["text"],
                    },
                )
            )

    # --------------------------------------------------------
    # Calcula diferenças
    # --------------------------------------------------------

    ids_obsoletos = (
        existing_ids
        - expected_ids
    )

    existentes = (
        expected_ids
        & existing_ids
    )

    novos_ids = (
        expected_ids
        - existing_ids
    )

    print()
    print("=" * 80)
    print("ANÁLISE")
    print("=" * 80)

    print(
        f"Embeddings no arquivo : {len(data)}"
    )

    print(
        f"IDs esperados         : "
        f"{len(expected_ids)}"
    )

    print(
        f"Já existentes         : "
        f"{len(existentes)}"
    )

    print(
        f"Novos                 : "
        f"{len(novos_ids)}"
    )

    print(
        f"Obsoletos              : "
        f"{len(ids_obsoletos)}"
    )

    # --------------------------------------------------------
    # Inserção incremental
    # --------------------------------------------------------

    if novos_points:

        print()
        print(
            f"Sincronizando "
            f"{len(novos_points)} "
            f"pontos novos..."
        )

        upsert_in_batches(
            client,
            novos_points
        )

    else:

        print()
        print(
            "Nenhum ponto novo para sincronizar."
        )

    # --------------------------------------------------------
    # Remove pontos obsoletos
    # --------------------------------------------------------

    if ids_obsoletos:

        delete_in_batches(
            client,
            list(ids_obsoletos)
        )

    else:

        print()
        print(
            "Nenhum ponto obsoleto."
        )

    # --------------------------------------------------------
    # Estado final
    # --------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 80)
    print("MIGRAÇÃO CONCLUÍDA")
    print("=" * 80)

    print(
        f"Embeddings          : {len(data)}"
    )

    print(
        f"Documentos           : {len(documentos)}"
    )

    print(
        f"Novos                : {len(novos_ids)}"
    )

    print(
        f"Já existentes        : {len(existentes)}"
    )

    print(
        f"Obsoletos removidos  : "
        f"{len(ids_obsoletos)}"
    )

    print(
        f"Points no Qdrant     : "
        f"{info.points_count}"
    )

    print("=" * 80)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()