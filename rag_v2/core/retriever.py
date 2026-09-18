import os
import json
import re
import requests
from typing import List, Dict, Set
from qdrant_client import QdrantClient
from .config import (
    QDRANT_URL, QDRANT_COLLECTION,
    TOP_K, FINAL_CONTEXTS, THRESHOLD,
    MAX_CHUNKS_PER_DOCUMENT,
    QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_COUNT,
    QUERY_EXPANSION_MODEL, OLLAMA_CHAT_URL,
    DATA_ROOT
)
from .embeddings import get_embedding

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client

_expansion_cache = {}
_corpus_doc_keywords = None

# ============================================================
# TAXONOMIA UNIVERSAL DE SISTEMAS OPERACIONAIS & INFRAESTRUTURA
# (Funciona de forma idêntica para qualquer SO: Linux, Windows, macOS, etc.)
# ============================================================

PLATFORM_KEYWORDS = {
    # Distribuições Linux & Ambientes Unix
    "linux", "rocky", "centos", "redhat", "rhel", "almalinux", "fedora",
    "debian", "ubuntu", "arch", "suse", "opensuse", "gentoo", "alpine",
    "unix", "bsd", "freebsd", "openbsd", "solaris",
    # Microsoft Windows
    "windows", "win10", "win11", "winserver", "windowsserver", "powershell", "wsl",
    # Apple macOS
    "macos", "osx", "darwin", "ios", "ipados",
    # Virtualização, Containers & Nuvem
    "docker", "podman", "kubernetes", "k8s", "vmware", "esxi", "proxmox",
    "hyper-v", "hyperv", "kvm", "xen", "aws", "azure", "gcp"
}

# Ações técnicas universais (em português e inglês)
ACTION_KEYWORDS = {
    # Ciclo de Vida e Instalação
    "instalação", "instalacao", "instalar", "instalado", "instalando", "setup", "deploy", "deployment",
    "atualização", "atualizacao", "atualizar", "upgrade", "update", "patch", "patching",
    "desinstalação", "desinstalacao", "desinstalar", "remoção", "remocao", "remover", "uninstall",
    # Configuração & Administração
    "configuração", "configuracao", "configurar", "ajuste", "ajustar", "tuning", "procedimento",
    "processo", "passo", "passos", "etapas", "requisitos", "requisito", "prerequisites", "pré-requisitos",
    "guia", "manual", "comandos", "comando", "sintaxe", "syntax", "script", "automatizar",
    # Problemas, Falhas e Diagnóstico
    "erro", "erros", "falha", "falhas", "bug", "crash", "travamento", "travar",
    "latência", "latencia", "lentidão", "lentidao", "timeout", "perda", "pacotes", "drop",
    "problema", "problemas", "solucionar", "solução", "solucao", "resolução", "resolucao",
    "troubleshooting", "diagnóstico", "diagnostico", "analisar", "verificar", "status", "healthcheck",
    # Operações de Disco, Rede e Permissões
    "particionamento", "particionar", "disco", "formatar", "formatação", "montar", "mount", "fstab",
    "boot", "inicialização", "inicializacao", "reiniciar", "reboot", "shutdown",
    "backup", "restauração", "restauracao", "restore", "recuperação", "recuperacao", "snapshot",
    "rede", "network", "porta", "portas", "ip", "subnet", "rota", "dns", "dhcp",
    "firewall", "iptables", "nftables", "ufw", "firewalld",
    "permissão", "permissao", "permissões", "permissoes", "chmod", "chown", "sudo", "root",
    "admin", "administrador", "login", "acesso", "sso", "ldap", "active directory", "ad",
    "ssl", "tls", "certificado"
}

# Termos de software/protocolos conhecidos do ambiente
CRITICAL_ENTITIES = {
    # Segurança & Antivírus
    "kaspersky", "kesl", "ksc", "antivirus", "antivírus", "clamav", "crowdstrike", "sentinelone",
    # Streaming & Acesso Remoto
    "pcoip", "anyware", "anywhere", "teradici", "teamviewer", "anydesk", "parsec", "rdp", "vnc",
    # Broadcast Áudio & IP
    "dante", "audinate", "rivage", "yamaha", "twinlane", "rio", "cl5", "ql5", "dm7",
    "aes67", "smpte", "st2110", "st2022", "ptp",
    # Broadcast Vídeo, Edição & Pós-Produção
    "flame", "autodesk", "maya", "3dsmax", "avid", "mediacentral", "mediacomposer",
    "interplay", "nexis", "isis", "davinci", "resolve", "nuke", "premiere", "aftereffects",
    # Armazenamento, Backup & Infraestrutura
    "bacula", "veeam", "networker", "commvault", "zabbix", "grafana", "prometheus",
    "nginx", "apache", "haproxy", "postfix", "samba", "sssd", "nfs", "iscsi", "lvm",
    # Pipeline IA & Web
    "qdrant", "ollama", "fastapi"
}

BRAND_KEYWORDS = CRITICAL_ENTITIES


def get_corpus_keywords() -> Set[str]:
    """
    Descobre dinamicamente palavras-chave a partir dos nomes de todos os documentos
    armazenados na base. Assim, qualquer novo PDF adicionado (ex: 'Manual_Veeam.pdf')
    tem seu assunto principal automaticamente catalogado sem precisar alterar código.
    """
    global _corpus_doc_keywords
    if _corpus_doc_keywords is not None:
        return _corpus_doc_keywords

    keywords = set()
    try:
        # 1. Tenta carregar do chunks.json se disponível
        chunks_path = os.path.join(DATA_ROOT, "database", "chunks.json")
        docs = set()
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            docs = {c.get("document", "") for c in chunks if c.get("document")}
        else:
            # 2. Tenta obter do Qdrant fazendo scroll de payloads
            client = get_qdrant_client()
            records, _ = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1000,
                with_payload=["document"],
                with_vectors=False
            )
            docs = {r.payload.get("document", "") for r in records if r.payload and r.payload.get("document")}

        stopwords_doc = {
            "manual", "guia", "guide", "livro", "book", "apostila", "pdf", "txt", "doc",
            "pt", "br", "en", "autor", "casa", "codigo", "volume", "vol", "edicao", "edition",
            "referencia", "reference", "user", "operation", "admin", "administracao", "overview"
        }
        for doc_name in docs:
            words = re.findall(r'[a-zA-Z0-9_\-]{3,}', doc_name.lower())
            for w in words:
                if w not in stopwords_doc and not w.isdigit():
                    keywords.add(w)
    except Exception:
        pass

    _corpus_doc_keywords = keywords
    return _corpus_doc_keywords


def analyze_query_components(query: str) -> Dict:
    """
    Decompõe a consulta de forma UNIVERSAL e DINÂMICA para qualquer software ou SO:
    - entities: Aplicações, ferramentas, protocolos ou termos substantivos específicos
    - platforms: Sistemas operacionais ou plataformas que funcionam como contexto (Linux, Windows, macOS, etc.)
    - actions: Intenções ou ações técnicas (instalação, configuração, erro, etc.)
    - clean_tokens: Todos os termos substantivos sem stopwords
    """
    # Normalização de erros comuns de digitação
    q_norm = re.sub(r'\banywhre\b', 'anywhere', query, flags=re.IGNORECASE)
    q_norm = re.sub(r'\banywere\b', 'anywhere', q_norm, flags=re.IGNORECASE)

    tokens = re.findall(r'[a-zA-Z0-9_\-\.]{2,}', q_norm)

    stopwords = {
        "como", "para", "posso", "pode", "qual", "quais", "onde", "quando", "quem",
        "por que", "porque", "esse", "essa", "este", "esta", "esses", "essas", "estes", "estas",
        "aquele", "aquela", "aqueles", "aquelas", "com", "sem", "por", "que", "uma", "umas",
        "um", "uns", "das", "dos", "nas", "nos", "sobre", "qualquer", "mais", "menos", "acessar",
        "fazer", "existe", "são", "sao", "nele", "dela", "dele", "ela", "ele", "eles", "elas",
        "seus", "suas", "meu", "minha", "você", "voce", "passos", "ter", "tendo", "sendo",
        "estar", "fica", "dá", "da", "do", "de", "no", "na", "em", "ao", "aos", "à", "às"
    }

    corpus_words = get_corpus_keywords()

    entities = []
    platforms = []
    actions = []
    other_terms = []

    for t in tokens:
        tl = t.lower()
        if tl in stopwords:
            continue
        if tl in PLATFORM_KEYWORDS:
            platforms.append(t)
        elif tl in ACTION_KEYWORDS:
            actions.append(t)
        elif tl in CRITICAL_ENTITIES or tl in corpus_words:
            entities.append(t)
        else:
            # Princípio Universal: Qualquer substantivo técnico que NÃO seja plataforma e NÃO seja ação
            # é, por definição, um assunto/software/ferramenta específica!
            # Ex: maya, nuke, sssd, lvm, bacula, zabbix, fortinet, davinci, etc.
            entities.append(t)

    clean_tokens = [t for t in tokens if t.lower() not in stopwords]

    return {
        "entities": entities,
        "platforms": platforms,
        "actions": actions,
        "other_terms": other_terms,
        "clean_tokens": clean_tokens
    }


def extract_technical_query(query: str) -> str:
    """Extrai termos técnicos principais da pergunta para busca concentrada."""
    comp = analyze_query_components(query)
    return " ".join(comp["clean_tokens"]) if comp["clean_tokens"] else query


def generate_search_queries(query: str) -> List[str]:
    """
    Gera representações complementares da intenção de busca para evitar Semantic Drift:
    1. Query Original completa: preserva sintaxe e nuances naturais.
    2. Query Canônica: termos limpos sem stopwords (ex: 'instalação Kaspersky Linux').
    3. Query Focada no Assunto / Software: Assunto + Ação (ex: 'Kaspersky instalação')
       -> Crucial: remove termos generalistas de alta frequência como 'Linux' ou 'Windows'
          para impedir que enciclopédias gerais sufoquem o manual específico do software.
    """
    comp = analyze_query_components(query)
    queries = [query]

    # 2. Query Canônica (termos essenciais limpos)
    canonical = " ".join(comp["clean_tokens"])
    if canonical and canonical.lower() != query.lower():
        queries.append(canonical)

    # 3. Query Focada no Assunto / Entidade (Universal Subject Focus)
    if comp["entities"]:
        # Se temos uma ferramenta ou software específico (ex: Maya, Nuke, Kaspersky, SSSD, LVM):
        # Montamos busca isolada: Entidade + Ações (sem a plataforma ampla que atrai livros gerais)
        entity_parts = comp["entities"] + comp["actions"]
        if comp["platforms"] and entity_parts:
            focused = " ".join(entity_parts)
            if focused.lower() not in [q.lower() for q in queries]:
                queries.append(focused)
    elif comp["platforms"] and comp["actions"]:
        # Quando a plataforma é o único sujeito (ex: 'particionar disco no Linux' ou 'firewall no Windows'):
        plat_focused = " ".join(comp["platforms"] + comp["actions"])
        if plat_focused.lower() not in [q.lower() for q in queries]:
            queries.append(plat_focused)

    # LLM Query Expansion (se habilitado nas configs)
    if QUERY_EXPANSION_ENABLED:
        cache_key = f"expand_{query}"
        if cache_key in _expansion_cache:
            return _expansion_cache[cache_key]

        prompt = f"""Gere {QUERY_EXPANSION_COUNT} versões alternativas da pergunta abaixo.
Mantenha o significado técnico. Uma por linha.

Pergunta: {query}"""

        try:
            response = requests.post(
                f"{OLLAMA_CHAT_URL}",
                json={
                    "model": QUERY_EXPANSION_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.2}
                },
                timeout=30
            )
            response.raise_for_status()

            content = response.json().get("message", {}).get("content", "")
            clean_expansions = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith(("-", "*", "•")):
                    line = line[1:].strip()
                elif len(line) > 2 and line[0].isdigit() and line[1] in (".", ")", ":", "-"):
                    line = line[2:].strip()
                if line.lower().startswith(("aqui estão", "aqui estao", "versão", "versao", "pergunta")):
                    continue
                if line and line != query and len(line) > 5:
                    clean_expansions.append(line)

            expansions = clean_expansions[:QUERY_EXPANSION_COUNT]
            result = queries + expansions if expansions else queries
            _expansion_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"⚠️ Query expansion LLM falhou: {e}")

    return queries


# Alias retrocompatível
expand_query = generate_search_queries


def retrieve(query: str, top_k: int = None, threshold: float = None) -> List[Dict]:
    """
    Recupera contextos relevantes através de busca vetorial multi-representação (Multi-Query),
    fusão RRF, desambiguação léxica com proteção de entidade e diversidade documental.
    """
    top_k = top_k if top_k is not None else TOP_K
    threshold = threshold if threshold is not None else THRESHOLD
    queries = generate_search_queries(query)

    all_candidates = {}
    client = get_qdrant_client()

    for current_query in queries:
        query_embedding = get_embedding(current_query, task_type="query").tolist()

        try:
            result = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            continue

        for rank, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            item_id = point.id

            if item_id not in all_candidates:
                all_candidates[item_id] = {
                    "id": item_id,
                    "score": float(point.score),
                    "rank": rank,
                    "document": payload.get("document"),
                    "page": payload.get("page"),
                    "chunk": payload.get("chunk"),
                    "text": payload.get("text", ""),
                    "rrf_ranks": [],
                    "appearances": 0,
                    "best_score": float(point.score),
                }

            candidate = all_candidates[item_id]
            candidate["rrf_ranks"].append(rank)
            candidate["appearances"] += 1
            candidate["best_score"] = max(candidate["best_score"], float(point.score))

    comp = analyze_query_components(query)

    # Identifica entidades críticas alvo presentes na pergunta:
    # 1. Se houver entidades de software/aplicação específicas (ex: Kaspersky, PCoIP), elas são prioritárias.
    # 2. Se não houver entidade de aplicação mas houver plataforma (ex: Rocky Linux), a plataforma vira alvo.
    target_entities = [e.lower() for e in comp["entities"]]
    if not target_entities and comp["platforms"]:
        target_entities = [p.lower() for p in comp["platforms"]]

    query_terms = [t.lower() for t in comp["clean_tokens"]]

    RRF_K = 60
    for item in all_candidates.values():
        item["rrf_score"] = sum(1.0 / (RRF_K + rank) for rank in item["rrf_ranks"])
        doc_str = (item.get("document") or "").lower()
        text_str = (item.get("text") or "").lower()

        # Proteção e Prioridade Absoluta de Entidade Crítica:
        # Se a pergunta citar 'Kaspersky' ou 'PCoIP', chunks que contêm essa entidade
        # ganham prioridade absoluta sobre chunks puramente contextuais (ex: só 'Linux').
        item["entity_match"] = any(e in doc_str or e in text_str for e in target_entities) if target_entities else False
        item["brand_match"] = item["entity_match"]  # Compatibilidade interna
        item["keyword_hits"] = sum(1 for term in query_terms if term in doc_str or term in text_str)

    candidates = [c for c in all_candidates.values() if c["best_score"] >= threshold]

    if not candidates and all_candidates:
        best_possible = max(c["best_score"] for c in all_candidates.values())
        print(f"⚠️ Nenhum candidato atingiu threshold={threshold:.2f} para '{query}'. Melhor score: {best_possible:.4f}")
        if best_possible >= 0.20:
            print(f"   ℹ️ Ativando fallback com threshold tolerante (>= 0.20)")
            candidates = [c for c in all_candidates.values() if c["best_score"] >= 0.20]
        else:
            return []
    elif not candidates:
        return []

    # ========================================================
    # ORDENAÇÃO PRIORITÁRIA HÍBRIDA:
    # 1. Chunks que contêm a entidade de aplicação específica (ex: Kaspersky, PCoIP)
    # 2. Chunks com mais termos coincidentes da pergunta (keyword hits)
    # 3. Score RRF (combinação ponderada das buscas original, canônica e focada)
    # 4. Número de aparições nas buscas
    # 5. Score bruto de similaridade
    # ========================================================
    candidates.sort(
        key=lambda x: (
            x.get("entity_match", False),
            x["keyword_hits"] > 0,
            x["keyword_hits"],
            x["rrf_score"],
            x["appearances"],
            x["best_score"]
        ),
        reverse=True
    )

    # ========================================================
    # DIVERSIDADE DOCUMENTAL
    # Impede que livros enciclopédicos gerais (ex: Linux a Bíblia)
    # ocupem mais de 2 slots, abrindo espaço para manuais específicos.
    # ========================================================
    selected = []
    doc_counts = {}

    for item in candidates:
        doc = item.get("document") or "desconhecido"
        if doc_counts.get(doc, 0) < MAX_CHUNKS_PER_DOCUMENT:
            selected.append(item)
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
            if len(selected) >= FINAL_CONTEXTS:
                break

    # Completa até FINAL_CONTEXTS se sobrarem slots
    if len(selected) < FINAL_CONTEXTS:
        selected_ids = {s["id"] for s in selected}
        for item in candidates:
            if item["id"] not in selected_ids:
                selected.append(item)
                if len(selected) >= FINAL_CONTEXTS:
                    break

    return [
        {
            "score": item["best_score"],
            "chunk": {
                "document": item["document"],
                "page": item["page"],
                "chunk": item["chunk"],
                "text": item["text"],
            }
        }
        for item in selected
    ]
