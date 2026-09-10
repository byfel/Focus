
"""
Popula a coleção rag_documents_v2 com os dados existentes
Usa os embeddings já gerados para não reprocessar tudo
"""

import sys
import json
import hashlib
from pathlib import Path

# Adiciona o path da versão atual
sys.path.insert(0, "/opt/ai/rag")
sys.path.insert(0, "/opt/ai/rag_v2")

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ============================================================
# CONFIGURAÇÃO
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_V2 = "rag_documents_v2"
EMBEDDINGS_FILE = "/opt/ai/rag/database/embeddings.json"
CHUNKS_FILE = "/opt/ai/rag/database/chunks.json"

def generate_qdrant_id(chunk_id):
    """Gera ID determinístico para o Qdrant"""
    value = int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16], 16)
    return value & 0x7FFFFFFFFFFFFFFF

def main():
    print("=" * 80)
    print("📊 POPULANDO COLEÇÃO rag_documents_v2")
    print("=" * 80)
    
    # 1. Carrega embeddings
    print("\n📂 Carregando embeddings...")
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    
    print(f"   ✅ {len(embeddings)} embeddings carregados")
    
    # 2. Conecta ao Qdrant
    print("\n🔌 Conectando ao Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Verifica coleção V2
    collections = client.get_collections().collections
    v2_exists = any(c.name == COLLECTION_V2 for c in collections)
    
    if not v2_exists:
        print(f"❌ Coleção {COLLECTION_V2} não encontrada!")
        print("   Execute o script de criação primeiro.")
        return
    
    # 3. Prepara pontos
    print("\n🔄 Preparando pontos para inserção...")
    points = []
    documentos = set()
    
    for item in embeddings:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            print(f"⚠️ Item sem chunk_id ignorado: {item.get('document')}")
            continue
        
        qdrant_id = generate_qdrant_id(chunk_id)
        documentos.add(item.get("document", "unknown"))
        
        points.append(
            PointStruct(
                id=qdrant_id,
                vector=item["embedding"],
                payload={
                    "chunk_id": chunk_id,
                    "document": item.get("document"),
                    "file_hash": item.get("file_hash"),
                    "page": item.get("page"),
                    "chunk": item.get("chunk"),
                    "text": item.get("text"),
                }
            )
        )
    
    print(f"   ✅ {len(points)} pontos preparados")
    print(f"   📄 {len(documentos)} documentos")
    
    # 4. Verifica pontos existentes
    print("\n🔍 Verificando pontos existentes...")
    existing_ids = set()
    offset = None
    
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_V2,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            existing_ids.add(record.id)
    
    print(f"   ✅ {len(existing_ids)} pontos existentes")
    
    # 5. Filtra apenas novos pontos
    new_points = [p for p in points if p.id not in existing_ids]
    print(f"   🆕 {len(new_points)} novos pontos para inserir")
    
    if not new_points:
        print("\n✅ Coleção já está atualizada!")
        return
    
    # 6. Insere em batches
    print("\n📥 Inserindo pontos...")
    batch_size = 100
    total = len(new_points)
    
    for i in range(0, total, batch_size):
        batch = new_points[i:i+batch_size]
        client.upsert(
            collection_name=COLLECTION_V2,
            points=batch,
            wait=True,
        )
        progress = min(i + batch_size, total)
        print(f"   📊 {progress}/{total} ({progress*100/total:.1f}%)")
    
    # 7. Verifica resultado
    info = client.get_collection(COLLECTION_V2)
    print(f"\n✅ Coleção populada com sucesso!")
    print(f"   📊 Total de pontos: {info.points_count}")
    print(f"   📄 Documentos: {len(documentos)}")
    
    # 8. Salva estado
    manifest = {
        "source": EMBEDDINGS_FILE,
        "collection": COLLECTION_V2,
        "points": info.points_count,
        "documents": list(documentos),
        "status": "populated"
    }
    
    with open("/opt/ai/rag_v2/data/population_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Manifesto salvo em: /opt/ai/rag_v2/data/population_manifest.json")

if __name__ == "__main__":
    main()
