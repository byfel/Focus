#!/usr/bin/env python3
"""
Script utilitário para iniciar o servidor web do RAG v2.

Uso:
    python3 scripts/run_web.py
    python3 scripts/run_web.py --port 8001
    python3 scripts/run_web.py --host 0.0.0.0 --port 8002
"""
import sys
import argparse
from pathlib import Path

# Adiciona rag_v2 ao path
V2_ROOT = Path(__file__).resolve().parent.parent
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from core.config import WEB_HOST, WEB_PORT, EMBEDDING_MODEL, QDRANT_COLLECTION


def main():
    parser = argparse.ArgumentParser(description="Inicia a interface Web do FOCUS AI RAG v2")
    parser.add_argument("--host", default=WEB_HOST, help=f"Host de escuta (padrão: {WEB_HOST})")
    parser.add_argument("--port", type=int, default=WEB_PORT, help=f"Porta HTTP (padrão: {WEB_PORT})")
    parser.add_argument("--reload", action="store_true", help="Ativa hot reload para desenvolvimento")
    args = parser.parse_args()

    print("=" * 70)
    print("🌐 INICIANDO SERVIDOR WEB • FOCUS AI RAG v2")
    print(f"   📡 URL: http://{args.host}:{args.port}")
    print(f"   ⚡ Embedding: {EMBEDDING_MODEL}")
    print(f"   📦 Coleção Qdrant: {QDRANT_COLLECTION}")
    print(f"   👤 Usuário de teste: admin / admin123")
    print("=" * 70)

    try:
        import uvicorn
        from api.server import app
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("❌ Uvicorn não está instalado.")
        print("   Instale via: pip install uvicorn fastapi python-multipart")
        sys.exit(1)


if __name__ == "__main__":
    main()
