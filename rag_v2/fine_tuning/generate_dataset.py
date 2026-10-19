#!/usr/bin/env python3
"""
FOCUS AI - Sintetizador Incremental de Dataset para Fine-Tuning de Solucionador de Mídia.

Transforma TODOS os manuais técnicos, KBs (Markdown/PDFs) e chunks (chunks.json / Qdrant)
em pares instrucionais de treinamento (ChatML / JSONL) estruturados sob a metodologia
rigorosa de troubleshooting de broadcast e pós-produção (TSHOOT / Root Cause Analysis).

Características principais:
- APRENDIZADO DE TODOS OS DOCUMENTOS: Agrupa por documento e seleciona trechos
  distribuídos ao longo de cada manual (Round-Robin entre documentos), evitando
  concentrar o treino em apenas um PDF.
- INCREMENTAL PARA NOVOS DOCUMENTOS: Rastreia o estado (SHA256 / file_hash e chunk_ids)
  em `dataset/dataset_state.json`. Em novas execuções após o `ingest.py`, processa
  automaticamente apenas os documentos novos, alterados ou que ainda não atingiram
  a cota de exemplos por documento.
- SALVAMENTO INCREMENTAL (CHECKPOINT): Grava cada exemplo no `.jsonl` e atualiza o
  estado imediatamente após cada chunk, permitindo pausar (`Ctrl+C`) e retomar sem perdas.

Uso:
    # 1. Processamento incremental padrão (aprende de todos os docs e foca nos novos):
    python3 generate_dataset.py --source chunks --chunks-per-doc 8

    # 2. Processar apenas documentos recém-ingeridos (ainda não presentes no estado):
    python3 generate_dataset.py --source chunks --only-new-docs

    # 3. Gerar apenas o dataset semente (Gold Standard) imediatamente (sem chamar LLM):
    python3 generate_dataset.py --source seeds-only
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Tenta carregar configurações do FOCUS v2 se disponível
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config import (
        OLLAMA_CHAT_URL, DEFAULT_LLM_MODEL, QDRANT_URL, QDRANT_COLLECTION,
        PROJECT_ROOT, DATA_ROOT
    )
except ImportError:
    OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemma3:12b")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents_v2")
    PROJECT_ROOT = os.getenv("RAG_V2_PROJECT_ROOT", "/opt/ai/rag_v2")
    DATA_ROOT = os.getenv("RAG_V2_DATA_ROOT", "/opt/ai/rag")

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_ft_dataset")

# ==============================================================================
# PROMPT DO SISTEMA: PERSONA DO SOLUCIONADOR DE MÍDIA
# ==============================================================================

SYSTEM_PROMPT = """Você é o FOCUS AI - Solucionador Sênior e Especialista em Engenharia de Mídia, Broadcast, Áudio/Vídeo Profissional, Infraestrutura e Automação de Estúdios.

Sua metodologia de resolução segue o protocolo estruturado de Troubleshooting Técnico:
1. SINTOMA: Descrição precisa do problema relatado.
2. FATOS_DOCUMENTADOS: Fatos comprovados e extraídos diretamente das evidências técnicas.
3. CAUSAS_DOCUMENTADAS / CAUSAS_DEDUZIVEIS: Relação técnica causal sem alucinar.
4. HIPOTESES: Possibilidades técnicas plausíveis quando a evidência exigir confirmação.
5. COMO_CONFIRMAR: Procedimentos práticos no terminal (Bash/Linux/PowerShell) ou interface (GUI) para validar a causa raiz.
6. ACAO_RECOMENDADA: Passo a passo claro com comandos exatos, scripts, parâmetros e caminhos de arquivo.
7. LIMITACOES: O que a evidência não permitiu aferir com certeza.

Ao criar scripts ou comandos (Bash, Python, automações):
- Forneça código pronto para produção, modular, com validação de erros e paths reais.
- Preserve equivalências de sistemas (ex: Rocky Linux / AlmaLinux / CentOS / RHEL).
Responda sempre em português técnico, objetivo, com alto rigor de engenharia."""

SYNTHESIZER_PROMPT_TEMPLATE = """Você é um Engenheiro Sênior de Mídia e IA encarregado de criar casos de treinamento supervisionado (Fine-Tuning) para um assistente técnico de estúdios e broadcast.

Documento de origem: {doc_name}
Abaixo está um trecho de DOCUMENTAÇÃO TÉCNICA / MANUAL / KB:
----------------------------------------
{context}
----------------------------------------

Sua tarefa:
Gere {count} caso(s) de suporte/troubleshooting ou procedimento operacional realista(s) baseados estritamente nas informações técnicas deste texto.
Cada caso deve conter:
1. Pergunta / Incidente do Usuário (Sintoma ou dúvida operacional relatada por um operador, montador de Flame/Avid/Resolve, ou engenheiro de broadcast sobre o sistema do documento '{doc_name}').
2. Resposta do Solucionador de Mídia, rigorosamente estruturada no formato:

SINTOMA:
<descrição do problema ou objetivo operacional>

FATOS_DOCUMENTADOS:
- <itens extraídos do contexto>

CAUSAS_DOCUMENTADAS:
- <causas ou requisitos técnicos explicados pelo texto>

COMO_CONFIRMAR:
1. <comandos de terminal (ex: ls -la, rpm, systemctl, netstat) ou inspeção na GUI>

ACAO_RECOMENDADA:
1. <passo a passo completo de resolução, configuração ou operação>

LIMITACOES:
- <ressalvas técnicas se houver>

IMPORTANTE: Responda APENAS em JSON válido no seguinte formato de lista:
[
  {{
    "user_query": "pergunta ou incidente do usuário",
    "assistant_response": "resposta técnica completa e formatada"
  }}
]
"""

# ==============================================================================
# SEEDS DE OURO (GOLD STANDARD DATASET)
# ==============================================================================

GOLD_STANDARD_SEEDS = [
    {
        "seed_id": "seed_sapphire_ofx_flame",
        "user_query": "Instalei o Boris FX Sapphire 2025 no Rocky Linux para o Autodesk Flame 2025, mas os nós (S_Glow, S_Blur) não aparecem no Batch. O que pode ser?",
        "assistant_response": """SINTOMA:
O Boris FX Sapphire 2025 não aparece no Autodesk Flame 2025 (módulos Batch e Node Bin) após a instalação no Rocky Linux.

FATOS_DOCUMENTADOS:
- A Boris FX descontinuou a arquitetura legada 'Sparks' a partir da versão 2020. No Flame 2025, o Sapphire opera exclusivamente como plugin OFX (Open FX).
- O pacote moderno oficial é o `SapphireOFX-2025.x.x86_64.rpm` e seu bundle reside em `/opt/BorisFX/SapphireOFX/Sapphire.ofx.bundle`.
- O diretório padrão onde o Flame busca plugins OFX é `/usr/OFX/Plugins/`.
- O Flame armazena em cache o estado dos nós em `ofxPluginCache.xml`.

CAUSAS_DOCUMENTADAS:
1. Symlink quebrado (broken symlink) em `/usr/OFX/Plugins/Sapphire.ofx.bundle`, frequentemente apontando para uma subpasta inexistente `/plugins/`.
2. Cache de plugins OFX do Flame desatualizado ou retendo estado de falha anterior.

COMO_CONFIRMAR:
1. Verifique a integridade do link simbólico global:
   ```bash
   ls -la /usr/OFX/Plugins/
   ```
   *Se o arquivo `Sapphire.ofx.bundle` aparecer em vermelho piscando, o link está quebrado (alvo inexistente).*
2. Confirme o caminho real do arquivo instalado pelo pacote RPM:
   ```bash
   rpm -ql SapphireOFX-2025* | grep "\\.bundle"
   ```

ACAO_RECOMENDADA:
1. Recrie o link simbólico apontando para o caminho físico real:
   ```bash
   ln -sfn /opt/BorisFX/SapphireOFX/Sapphire.ofx.bundle /usr/OFX/Plugins/Sapphire.ofx.bundle
   ```
2. Valide se o link agora aparece em azul/ciano:
   ```bash
   ls -la /usr/OFX/Plugins/Sapphire.ofx.bundle
   ```
3. Limpe o cache de plugins OFX do Flame para forçar novo scan no próximo boot:
   ```bash
   rm -f /opt/Autodesk/flame_*/status/ofxPluginCache.xml
   ```
4. Reinicie o Flame e faça a busca por `S_Glow` no Batch (atalho Espaço).

LIMITACOES:
- Se houver conflito de licença (RLM Server Boris FX), o plugin pode carregar com marca d'água (X vermelho) em vez de ficar invisível."""
    },
    {
        "seed_id": "seed_dante_controller_red",
        "user_query": "No Dante Controller, o nome de um dos nossos conversores de áudio está aparecendo em texto vermelho e não conseguimos rotear nenhum canal para ele. O que significa?",
        "assistant_response": """SINTOMA:
Nome do dispositivo exibido em texto vermelho no Dante Controller, impedindo o roteamento e a subscrição de canais de áudio sobre IP.

FATOS_DOCUMENTADOS:
- No Dante Controller, a cor do texto do dispositivo reflete o estado de conectividade de rede e compatibilidade de endereçamento IP.
- Texto em vermelho indica inconsistência de sub-rede IP (Subnet Mismatch) entre o dispositivo Dante e a placa de rede do computador de controle/roteamento.

CAUSAS_DOCUMENTADAS:
1. O dispositivo obteve um endereço IP em uma sub-rede diferente (ex: IP estático antigo fora do escopo ou fallback link-local 169.254.x.x enquanto o switch/controlador está em 192.168.x.x ou 10.x.x.x).
2. Falha de concessão do servidor DHCP dedicado da rede de áudio.

COMO_CONFIRMAR:
1. No Dante Controller, dê um duplo clique no dispositivo em vermelho para abrir a janela 'Device View'.
2. Acesse a aba 'Configure' ou 'Network Config'.
3. Compare o IP e Máscara de Sub-rede do dispositivo com o IP da interface de rede (NIC) da sua máquina de monitoramento.

ACAO_RECOMENDADA:
1. Configure temporariamente a interface do seu computador com um IP secundário na mesma faixa do dispositivo em vermelho para restabelecer comunicação.
2. Na janela 'Device View' do Dante Controller, acesse 'Network Config' e altere a configuração do dispositivo para:
   - 'Obtain an IP Address Automatically' (DHCP) caso sua infraestrutura utilize DHCP; OU
   - Defina um IP estático válido pertencente à sub-rede oficial da rede Dante de produção.
3. Clique em 'Apply' e reinicie o equipamento de áudio caso solicitado pelo controlador.
4. Reabra a matriz de roteamento e valide se o dispositivo passou a exibir o nome em preto normal e com ícones de status verde (Subscription OK).

LIMITACOES:
- Se a rede utilizar redundância Dante (Primary e Secondary), certifique-se de que os cabos não foram invertidos entre os switches Primary e Secondary."""
    },
    {
        "seed_id": "seed_pcoip_disconnects_linux",
        "user_query": "Nossos clientes PCoIP (HP Anyware) estão sofrendo desconexões aleatórias a cada 15 minutos ao conectar nas ilhas de edição Linux. Como investigar no terminal?",
        "assistant_response": """SINTOMA:
Sessões de vídeo PCoIP (HP Anyware / Teradici) desconectando intermitentemente em estações de trabalho de pós-produção Linux.

FATOS_DOCUMENTADOS:
- O agente PCoIP utiliza as portas UDP 4172 (tráfego de mídia/vídeo/áudio) e TCP 4172 / TCP 443 (autenticação e handshake de sessão).
- O log detalhado de execução do agente PCoIP no Linux é gravado por padrão em `/var/log/pcoip-agent/`.

CAUSAS_DOCUMENTADAS:
1. MTU inadequado causando fragmentação de pacotes UDP na rede VPN/WAN.
2. Timeouts de firewall inspecionando pacotes UDP da porta 4172 ou esgotamento de tabela NAT.
3. Queda de handshake TLS no daemon de gerenciamento do agente.

COMO_CONFIRMAR:
1. Inspecione os logs mais recentes do PCoIP Agent no host Linux:
   ```bash
   grep -Ei "disconnect|session dropped|loss|timeout" /var/log/pcoip-agent/*.log | tail -n 50
   ```
2. Verifique se as portas 4172 UDP e TCP estão abertas e em escuta:
   ```bash
   ss -tulnp | grep 4172
   ```
3. Teste perda de pacotes e MTU entre o cliente e o host:
   ```bash
   ping -M do -s 1400 -c 20 <IP_DO_CLIENTE_OU_HOST>
   ```

ACAO_RECOMENDADA:
1. Caso haja fragmentação na rede externa, ajuste a política de MTU do PCoIP:
   Edite o arquivo `/etc/pcoip-agent/pcoip-agent.conf`:
   ```ini
   pcoip.mtu_size = 1300
   ```
2. Assegure que as regras de firewall no host (firewalld / iptables) permitam tráfego contínuo nas portas:
   ```bash
   firewall-cmd --permanent --add-port=4172/tcp
   firewall-cmd --permanent --add-port=4172/udp
   firewall-cmd --reload
   ```
3. Reinicie o serviço do agente PCoIP:
   ```bash
   systemctl restart pcoip
   ```

LIMITACOES:
- Se houver switches gerenciáveis com QoS desabilitado na rota, picos de tráfego de armazenamento (SAN/NAS) podem descartar pacotes UDP prioritários do PCoIP."""
    },
    {
        "seed_id": "seed_wiretap_watchdog_bash",
        "user_query": "Preciso de um script em Bash para rodar no servidor Rocky Linux que valide se o Wiretap do Flame está ativo, teste as portas de rede e reinicie o serviço se necessário.",
        "assistant_response": """SINTOMA:
Necessidade de automação em Bash para monitoramento, validação de portas e recuperação automática do serviço Autodesk Wiretap no Rocky Linux / CentOS.

FATOS_DOCUMENTADOS:
- O Autodesk Wiretap gerencia a transferência de mídia, render em rede e interoperabilidade do Flame.
- Os serviços essenciais do ecossistema Autodesk Wiretap incluem `adsk_wiretap`, `adsk_wiretapgateway` e o daemon do Stone+Wire (`adsk_stone_wire`).
- O Wiretap opera primariamente nas portas TCP 7100 a 7104.

COMO_CONFIRMAR:
O script abaixo realiza a checagem das portas, estado do daemon systemd e reinicialização segura:

ACAO_RECOMENDADA:
Crie o script `/usr/local/bin/check_wiretap.sh`:

```bash
#!/usr/bin/env bash
# ==============================================================================
# FOCUS AI - Monitor e Watchdog do Autodesk Wiretap & Stone+Wire
# Compatibilidade: Rocky Linux 8/9, CentOS 7/8, RHEL
# ==============================================================================
set -euo pipefail

LOG_FILE="/var/log/focus_wiretap_monitor.log"
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_FILE"
}

check_service() {
    local svc="$1"
    if systemctl is-active --quiet "$svc"; then
        log "✅ Serviço $svc está ATIVO."
        return 0
    else
        log "⚠️  ALERTA: Serviço $svc está PARADO ou com falha. Tentando reiniciar..."
        systemctl restart "$svc"
        sleep 2
        if systemctl is-active --quiet "$svc"; then
            log "✅ Serviço $svc recuperado com sucesso!"
            return 0
        else
            log "❌ ERRO CRÍTICO: Falha ao reiniciar $svc."
            return 1
        fi
    fi
}

log "--- Iniciando Diagnóstico do Autodesk Wiretap ---"

# 1. Checagem dos serviços systemd da Autodesk
SERVICES=("adsk_stone_wire" "adsk_wiretap" "adsk_wiretapgateway")
ALL_OK=true

for s in "${SERVICES[@]}"; do
    if systemctl list-unit-files | grep -q "^$s.service"; then
        check_service "$s" || ALL_OK=false
    fi
done

# 2. Checagem da porta TCP 7100 (Wiretap Server)
if ss -tuln | grep -q ":7100 "; then
    log "✅ Porta TCP 7100 (Wiretap Daemon) respondendo normalmente."
else
    log "❌ AVISO: Nenhuma escuta na porta 7100. Verifique as configurações em /opt/Autodesk/wiretap/."
    ALL_OK=false
fi

if [ "$ALL_OK" = true ]; then
    log "🎯 Todos os subsistemas de mídia do Wiretap estão operacionais."
    exit 0
else
    log "⚠️  Intervenção manual necessária nos logs em /opt/Autodesk/log/."
    exit 1
fi
```

Torne o script executável:
```bash
chmod +x /usr/local/bin/check_wiretap.sh
```

LIMITACOES:
- Se a partição do Stone+Wire (`/mnt/StorageMedia`) estiver corrompida ou sem espaço (100% full), o `adsk_stone_wire` falhará mesmo após o restart."""
    }
]


# ==============================================================================
# GERENCIAMENTO DE ESTADO INCREMENTAL E PERSISTÊNCIA
# ==============================================================================

def load_dataset_state(state_file: Path) -> Dict:
    """Carrega o estado incremental de geração de dataset."""
    default_state = {
        "processed_chunk_ids": {},  # chunk_id -> {"document": str, "samples": int, "timestamp": str}
        "documents": {},            # doc_name -> {"file_hash": str, "chunks_processed": int, "samples_generated": int, "updated_at": str}
        "seeds_added": False
    }
    if not state_file.exists():
        return default_state
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in default_state.items():
                if k not in data:
                    data[k] = v
            return data
    except Exception as e:
        logger.warning(f"Erro ao ler state file {state_file}: {e}")
        return default_state


def save_dataset_state(state_file: Path, state: Dict):
    """Salva o estado incremental de geração de dataset."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = state_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp_file.replace(state_file)


def load_existing_jsonl(output_file: Path) -> Tuple[List[Dict], set]:
    """
    Carrega exemplos já existentes no arquivo .jsonl para não perder progresso anterior
    e retorna (lista_de_exemplos, set_de_user_queries).
    """
    samples = []
    seen_queries = set()
    if not output_file.exists():
        return samples, seen_queries

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                msgs = item.get("messages", [])
                user_q = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                if user_q and user_q not in seen_queries:
                    seen_queries.add(user_q)
                    samples.append(item)
    except Exception as e:
        logger.warning(f"Erro ao ler {output_file} existente: {e}")

    return samples, seen_queries


def append_samples_to_jsonl(output_file: Path, new_samples: List[Dict]):
    """Adiciona novos exemplos incrementalmente ao final do arquivo .jsonl."""
    if not new_samples:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for item in new_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def rewrite_jsonl(output_file: Path, all_samples: List[Dict]):
    """Reescreve o arquivo .jsonl completo de forma atômica."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_file.replace(output_file)


# ==============================================================================
# SELEÇÃO INTELIGENTE DE CHUNKS (TODOS OS DOCUMENTOS + PRIORIDADE A NOVOS)
# ==============================================================================

def score_chunk_relevance(text: str) -> float:
    """
    Pontua a densidade técnica de um chunk para priorizar páginas de conteúdo
    técnico real (comandos, procedimentos, configurações, erros) em vez de capas/índices.
    """
    score = min(len(text) / 500.0, 3.0)
    tech_Markers = [
        r"/opt/", r"/etc/", r"/var/", r"/usr/", r"systemctl", r"chmod", r"ln -s",
        r"erro", r"error", r"falha", r"configura", r"procedimento", r"passo",
        r"menu", r"botão", r"porta", r"ip", r"rede", r"plugin", r"servidor",
        r"instala", r"comando", r"terminal", r"script", r"status", r"log"
    ]
    for pattern in tech_Markers:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.6
    # Penaliza sumários pontilhados (ex: "1. Introdução ............ 5")
    if len(re.findall(r"\.{4,}", text)) >= 2:
        score -= 3.0
    return score


def select_representative_chunks_for_doc(doc_chunks: List[Dict], max_per_doc: int) -> List[Dict]:
    """
    Seleciona até `max_per_doc` chunks distribuídos ao longo do documento,
    combinando cobertura espacial (início, meio e fim do documento) e densidade técnica.
    Se `max_per_doc <= 0` ou `len(doc_chunks) <= max_per_doc`, retorna todos os chunks válidos.
    """
    if max_per_doc <= 0 or len(doc_chunks) <= max_per_doc:
        return doc_chunks

    # Filtra chunks com score muito baixo (ex: índice puro) se houver suficientes
    scored = [(score_chunk_relevance(c["text"]), idx, c) for idx, c in enumerate(doc_chunks)]
    valid_scored = [item for item in scored if item[0] > 0]
    pool = [item[2] for item in (valid_scored if len(valid_scored) >= max_per_doc else scored)]

    if len(pool) <= max_per_doc:
        return pool

    # Divide o documento em `max_per_doc` faixas (buckets) e escolhe o chunk mais rico de cada faixa
    selected = []
    bucket_size = len(pool) / float(max_per_doc)
    for i in range(max_per_doc):
        start_idx = int(round(i * bucket_size))
        end_idx = int(round((i + 1) * bucket_size))
        bucket = pool[start_idx:max(end_idx, start_idx + 1)]
        if bucket:
            best_in_bucket = max(bucket, key=lambda c: score_chunk_relevance(c["text"]))
            selected.append(best_in_bucket)

    return selected


def load_all_chunks_from_file(chunks_file: Path) -> List[Dict]:
    """Lê todos os chunks válidos de chunks.json (gerado pelo ingest.py)."""
    if not chunks_file.exists():
        logger.warning(f"Arquivo chunks.json não encontrado em: {chunks_file}")
        return []
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        chunks = []
        for idx, item in enumerate(items):
            text = (item.get("text") or item.get("content") or "").strip()
            if len(text) < 200:
                continue
            doc_name = item.get("document", "documento_desconhecido")
            page = item.get("page", 1)
            chunk_num = item.get("chunk", idx + 1)
            chunk_id = item.get("chunk_id") or hashlib.sha256(
                f"{doc_name}|{page}|{chunk_num}|{text}".encode("utf-8")
            ).hexdigest()
            file_hash = item.get("file_hash") or ""
            chunks.append({
                "chunk_id": chunk_id,
                "name": doc_name,
                "file_hash": file_hash,
                "page": page,
                "chunk": chunk_num,
                "source_type": item.get("source_type", "text"),
                "text": text
            })
        return chunks
    except Exception as e:
        logger.error(f"Erro ao carregar chunks de {chunks_file}: {e}")
        return []


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 20) -> str:
    """Extrai texto de arquivo PDF usando PyMuPDF (fitz) se disponível."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for i in range(min(len(doc), max_pages)):
            page_text = doc[i].get_text()
            if page_text.strip():
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.debug(f"Não foi possível extrair PDF {pdf_path}: {e}")
        return ""


def load_documents_from_dir(docs_dir: Path) -> List[Dict]:
    """Lê arquivos markdown, texto ou PDFs locais que não estejam em chunks.json."""
    chunks = []
    if not docs_dir.exists():
        return chunks

    for file_path in sorted(docs_dir.glob("**/*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        content = ""
        if ext in [".md", ".txt"]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                logger.error(f"Erro ao ler {file_path}: {e}")
        elif ext == ".pdf":
            content = extract_text_from_pdf(file_path)

        if len(content) > 200:
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            # Divide textos grandes em blocos de ~2500 caracteres
            blocks = [content[i:i + 2500] for i in range(0, len(content), 2500)]
            for b_idx, block in enumerate(blocks, start=1):
                if len(block.strip()) < 150:
                    continue
                chunk_id = hashlib.sha256(f"{file_path.name}|{b_idx}|{block}".encode("utf-8")).hexdigest()
                chunks.append({
                    "chunk_id": chunk_id,
                    "name": file_path.name,
                    "file_hash": file_hash,
                    "page": b_idx,
                    "chunk": b_idx,
                    "source_type": "file",
                    "text": block
                })
    return chunks


def load_chunks_from_qdrant(limit: int = 5000) -> List[Dict]:
    """Scrolla chunks diretamente do Qdrant da base do FOCUS."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=15)
        chunks = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=min(1000, limit - len(chunks)),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for r in records:
                p = r.payload or {}
                text = p.get("text", "").strip()
                if len(text) > 200:
                    doc_name = p.get("document", "qdrant_chunk")
                    page = p.get("page", 1)
                    chunk_num = p.get("chunk", 1)
                    chunk_id = p.get("chunk_id") or hashlib.sha256(
                        f"{doc_name}|{page}|{chunk_num}|{text}".encode("utf-8")
                    ).hexdigest()
                    chunks.append({
                        "chunk_id": chunk_id,
                        "name": doc_name,
                        "file_hash": p.get("file_hash", ""),
                        "page": page,
                        "chunk": chunk_num,
                        "source_type": p.get("source_type", "text"),
                        "text": text
                    })
            if offset is None or len(chunks) >= limit:
                break
        return chunks
    except Exception as e:
        logger.warning(f"Não foi possível conectar ao Qdrant ({QDRANT_URL}): {e}")
        return []


def plan_incremental_work(
    all_chunks: List[Dict],
    state: Dict,
    chunks_per_doc: int = 8,
    max_samples: int = 0,
    only_new_docs: bool = False,
    skip_docs: Optional[List[str]] = None
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Agrupa todos os chunks por documento, analisa o estado incremental (igual ao ingest.py)
    e constrói a fila de processamento intercalada (Round-Robin) priorizando documentos novos!
    """
    skip_set = set(skip_docs or [])
    by_doc: Dict[str, List[Dict]] = {}
    for ch in all_chunks:
        by_doc.setdefault(ch["name"], []).append(ch)

    processed_chunk_ids = state.get("processed_chunk_ids", {})
    doc_states = state.get("documents", {})

    doc_status_report: Dict[str, str] = {}
    new_docs_queues: Dict[str, List[Dict]] = {}
    partial_docs_queues: Dict[str, List[Dict]] = {}

    print("\n" + "=" * 75)
    print("📋 ANÁLISE INCREMENTAL DE DOCUMENTOS PARA FINE-TUNING")
    print("=" * 75)

    for doc_name in sorted(by_doc.keys()):
        doc_chunks = by_doc[doc_name]
        current_hash = next((c["file_hash"] for c in doc_chunks if c.get("file_hash")), "")

        if doc_name in skip_set:
            doc_status_report[doc_name] = "IGNORADO (--skip-docs)"
            continue

        prev_doc_info = doc_states.get(doc_name)
        prev_hash = prev_doc_info.get("file_hash", "") if prev_doc_info else ""
        already_done_for_doc = sum(
            1 for c in doc_chunks if c["chunk_id"] in processed_chunk_ids
        )
        # Se foi marcado manualmente ou via estado anterior
        recorded_done = max(
            already_done_for_doc,
            prev_doc_info.get("chunks_processed", 0) if (prev_doc_info and prev_hash == current_hash) else 0
        )

        # Determina meta de chunks para este documento
        target_for_doc = len(doc_chunks) if chunks_per_doc <= 0 else min(chunks_per_doc, len(doc_chunks))

        # Caso 1: Documento nunca visto antes
        if prev_doc_info is None and already_done_for_doc == 0:
            selected = select_representative_chunks_for_doc(doc_chunks, target_for_doc)
            new_docs_queues[doc_name] = selected
            doc_status_report[doc_name] = f"🆕 NOVO DOCUMENTO ({len(selected)}/{len(doc_chunks)} chunks selecionados)"

        # Caso 2: Documento foi alterado (hash do PDF mudou no ingest.py)
        elif prev_hash and current_hash and prev_hash != current_hash:
            unprocessed = [c for c in doc_chunks if c["chunk_id"] not in processed_chunk_ids]
            selected = select_representative_chunks_for_doc(unprocessed, target_for_doc)
            new_docs_queues[doc_name] = selected
            doc_status_report[doc_name] = f"♻️  DOCUMENTO ALTERADO ({len(selected)} novos chunks selecionados)"

        # Caso 3: Documento já atingiu a cota de aprendizado e está inalterado
        elif recorded_done >= target_for_doc:
            doc_status_report[doc_name] = f"✅ JÁ APRENDIDO ({recorded_done}/{target_for_doc} chunks processados)"

        # Caso 4: Documento parcialmente processado (ainda faltam chunks para atingir a meta)
        else:
            if only_new_docs:
                doc_status_report[doc_name] = f"⏭️  PARCIALMENTE APRENDIDO ({recorded_done}/{target_for_doc} - pulado por --only-new-docs)"
            else:
                remaining_needed = target_for_doc - recorded_done
                unprocessed = [c for c in doc_chunks if c["chunk_id"] not in processed_chunk_ids]
                selected = select_representative_chunks_for_doc(unprocessed, remaining_needed)
                if selected:
                    partial_docs_queues[doc_name] = selected
                    doc_status_report[doc_name] = f"⏳ COMPLEMENTANDO ({recorded_done} feitos, +{len(selected)} chunks)"
                else:
                    doc_status_report[doc_name] = f"✅ JÁ APRENDIDO ({recorded_done} chunks)"

    for doc_name, status_str in doc_status_report.items():
        print(f"  • {doc_name}: {status_str}")

    # Intercala em Round-Robin: primeiro TODOS os documentos NOVOS/ALTERADOS,
    # depois os documentos parciais (garantindo que nenhum documento fique de fora!)
    ordered_queue: List[Dict] = []

    def interleave_queues(queues_dict: Dict[str, List[Dict]]) -> List[Dict]:
        result = []
        active_docs = sorted(queues_dict.keys())
        round_idx = 0
        while True:
            added_in_round = False
            for d_name in active_docs:
                q = queues_dict[d_name]
                if round_idx < len(q):
                    result.append(q[round_idx])
                    added_in_round = True
            if not added_in_round:
                break
            round_idx += 1
        return result

    ordered_queue.extend(interleave_queues(new_docs_queues))
    ordered_queue.extend(interleave_queues(partial_docs_queues))

    if max_samples and max_samples > 0:
        ordered_queue = ordered_queue[:max_samples]

    print("-" * 75)
    print(f"  Total de documentos na base       : {len(by_doc)}")
    print(f"  Documentos novos/alterados        : {len(new_docs_queues)}")
    print(f"  Documentos complementares         : {len(partial_docs_queues)}")
    print(f"  Chunks enfileirados nesta rodada  : {len(ordered_queue)}")
    print("=" * 75 + "\n")

    return ordered_queue, doc_status_report


# ==============================================================================
# SÍNTESE VIA OLLAMA
# ==============================================================================

def generate_cases_from_text(
    context_text: str,
    doc_name: str,
    model: str = DEFAULT_LLM_MODEL,
    count: int = 2
) -> List[Dict]:
    """Envia o trecho do manual para o Ollama e gera casos de suporte estruturados."""
    prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(
        doc_name=doc_name,
        context=context_text[:3000],
        count=count
    )

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Você é um assistente especialista que gera datasets técnicos estritamente em JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=180
        )
        response.raise_for_status()
        res_data = response.json()
        raw_content = res_data.get("message", {}).get("content", "").strip()

        # Extrai JSON de blocos de markdown ```json ... ``` se existirem
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "user_query" in parsed:
            return [parsed]
        return []
    except Exception as e:
        logger.debug(f"Falha ao sintetizar casos para {doc_name}: {e}")
        return []


def format_to_chatml(system_prompt: str, user_query: str, assistant_response: str) -> Dict:
    """Formata o exemplo no padrão ChatML (padrão ouro para SFTTrainer / Hugging Face)."""
    return {
        "messages": [
            {"role": "system", "content": str(system_prompt).strip()},
            {"role": "user", "content": str(user_query).strip()},
            {"role": "assistant", "content": str(assistant_response).strip()}
        ]
    }


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="FOCUS AI - Gerador Incremental de Dataset para Fine-Tuning")
    parser.add_argument("--source", choices=["files", "chunks", "qdrant", "seeds-only", "all"], default="chunks",
                        help="Fonte: 'chunks' (chunks.json da V1/V2), 'files' (diretório de PDFs/MDs), 'qdrant', 'seeds-only' ou 'all'")
    parser.add_argument("--docs-dir", type=str, default=None,
                        help="Diretório de documentos/manuais (ex: /opt/ai/rag/documents)")
    parser.add_argument("--chunks-file", type=str, default=None,
                        help="Caminho para chunks.json (padrão: /opt/ai/rag/database/chunks.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Caminho do arquivo .jsonl gerado")
    parser.add_argument("--state-file", type=str, default=None,
                        help="Caminho do arquivo de estado incremental (padrão: dataset/dataset_state.json)")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                        help="Modelo do Ollama para geração sintética")
    parser.add_argument("--chunks-per-doc", type=int, default=8,
                        help="Quantidade de chunks representativos por documento (0 = todos os chunks do documento). Padrão: 8")
    parser.add_argument("--cases-per-chunk", type=int, default=2,
                        help="Quantidade de casos de pergunta/resposta gerados por chunk. Padrão: 2")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limite máximo de chunks a processar nesta execução (0 = processar todos os documentos planejados)")
    parser.add_argument("--only-new-docs", action="store_true",
                        help="Processar apenas documentos novos ou alterados que ainda não estão no dataset_state.json")
    parser.add_argument("--mark-learned", type=str, nargs="*", default=None,
                        help="Lista de nomes de documentos já aprendidos para registrar no estado sem reprocessar (ex: 'Criação de Produto no Diva V2.pdf')")
    parser.add_argument("--skip-docs", type=str, nargs="*", default=None,
                        help="Lista de documentos para ignorar nesta execução")
    parser.add_argument("--recreate", action="store_true",
                        help="Apaga o estado e recria o dataset do zero")
    parser.add_argument("--status", action="store_true",
                        help="Apenas exibe o relatório de cobertura de todos os documentos (sem chamar o LLM)")
    parser.add_argument("--include-seeds", action="store_true", default=True,
                        help="Incluir exemplos sementes de alta qualidade (padrão: True)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output) if args.output else output_dir / "media_solver_train.jsonl"
    state_file = Path(args.state_file) if args.state_file else output_dir / "dataset_state.json"

    if args.recreate:
        logger.info("♻️  Modo --recreate ativado: reiniciando dataset e estado incremental...")
        if output_file.exists():
            output_file.unlink()
        if state_file.exists():
            state_file.unlink()

    state = load_dataset_state(state_file)
    existing_samples, seen_queries = load_existing_jsonl(output_file)

    if existing_samples:
        logger.info(f"📂 Dataset existente carregado: {len(existing_samples)} exemplos preservados em {output_file.name}")

    # 1. Garante que as Seeds de Ouro estejam presentes sem duplicar
    if not args.status and (args.include_seeds or args.source == "seeds-only"):
        new_seed_samples = []
        for seed in GOLD_STANDARD_SEEDS:
            if seed["user_query"] not in seen_queries:
                chatml_sample = format_to_chatml(
                    system_prompt=SYSTEM_PROMPT,
                    user_query=seed["user_query"],
                    assistant_response=seed["assistant_response"]
                )
                new_seed_samples.append(chatml_sample)
                existing_samples.append(chatml_sample)
                seen_queries.add(seed["user_query"])

        if new_seed_samples:
            logger.info(f"Adicionando {len(new_seed_samples)} casos sementes de alta fidelidade...")
            append_samples_to_jsonl(output_file, new_seed_samples)
        state["seeds_added"] = True
        save_dataset_state(state_file, state)

    if args.source == "seeds-only" and not args.status:
        logger.info(f"✅ Concluído! Total de exemplos no dataset: {len(existing_samples)} ({output_file})")
        return

    # 2. Carrega todos os chunks disponíveis de todas as fontes solicitadas
    all_chunks = []

    # Opção A: chunks.json (gerado pelo ingest.py)
    if args.source in ["chunks", "all"]:
        candidate_chunks = [
            Path(args.chunks_file) if args.chunks_file else None,
            Path(DATA_ROOT) / "database" / "chunks.json",
            Path(PROJECT_ROOT) / "data" / "database" / "chunks.json",
            Path("/opt/ai/rag/database/chunks.json"),
        ]
        chosen_chunks = next((p for p in candidate_chunks if p and p.exists()), None)
        if chosen_chunks:
            logger.info(f"Carregando base completa de chunks: {chosen_chunks}")
            all_chunks.extend(load_all_chunks_from_file(chosen_chunks))
        else:
            logger.warning("Nenhum chunks.json encontrado nos caminhos padrão.")

    # Opção B: arquivos diretos (PDFs, Markdown, etc. em data/documents)
    if args.source in ["files", "all"]:
        candidate_dirs = []
        if args.docs_dir:
            candidate_dirs.append(Path(args.docs_dir))
        else:
            candidate_dirs.extend([
                Path(__file__).resolve().parent.parent / "data" / "documents",
                Path(PROJECT_ROOT) / "data" / "documents",
                Path(DATA_ROOT) / "documents",
                Path("/opt/ai/rag/documents"),
            ])

        known_doc_names = {c["name"] for c in all_chunks}
        scanned_dirs = set()
        for d in candidate_dirs:
            if d.exists() and str(d.resolve()) not in scanned_dirs:
                scanned_dirs.add(str(d.resolve()))
                logger.info(f"Verificando documentos locais em: {d}")
                dir_chunks = load_documents_from_dir(d)
                # Adiciona documentos locais que ainda não vieram do chunks.json (ex: .md de TSHOOT)
                for dc in dir_chunks:
                    if args.source == "files" or dc["name"] not in known_doc_names:
                        all_chunks.append(dc)

    # Opção C: Qdrant
    if args.source in ["qdrant", "all"] and not all_chunks:
        logger.info(f"Buscando chunks do Qdrant ({QDRANT_COLLECTION})...")
        all_chunks.extend(load_chunks_from_qdrant(limit=10000))

    if not all_chunks:
        logger.warning("Nenhum documento/chunk encontrado para sintetizar.")
        return

    # 3. Se o usuário passou --mark-learned, registra esses documentos como já aprendidos no estado
    if args.mark_learned:
        for doc_name in args.mark_learned:
            matching = [c for c in all_chunks if c["name"] == doc_name]
            file_hash = next((c["file_hash"] for c in matching if c.get("file_hash")), "")
            state["documents"][doc_name] = {
                "file_hash": file_hash,
                "chunks_processed": len(matching) if matching else 99,
                "samples_generated": state["documents"].get(doc_name, {}).get("samples_generated", 0),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"📌 Documento marcado como já aprendido: {doc_name}")
        save_dataset_state(state_file, state)

    # 4. Planeja a fila incremental distribuída entre TODOS os documentos (priorizando novos)
    queue, status_report = plan_incremental_work(
        all_chunks=all_chunks,
        state=state,
        chunks_per_doc=args.chunks_per_doc,
        max_samples=args.max_samples,
        only_new_docs=args.only_new_docs,
        skip_docs=args.skip_docs
    )

    if args.status:
        learned_count = sum(1 for s in status_report.values() if "JÁ APRENDIDO" in s)
        total_docs = len(status_report)
        train_state_file = Path(__file__).resolve().parent / "output" / "train_state.json"
        print("📊 RESUMO GERAL DE COBERTURA DO TREINAMENTO:")
        print(f"   • Documentos 100% aprendidos no dataset : {learned_count} / {total_docs}")
        print(f"   • Documentos pendentes / novos          : {total_docs - learned_count}")
        print(f"   • Total de exemplos no dataset (.jsonl) : {len(existing_samples)}")
        if train_state_file.exists():
            try:
                ts = json.loads(train_state_file.read_text(encoding="utf-8"))
                current_sha = hashlib.sha256(output_file.read_bytes()).hexdigest() if output_file.exists() else ""
                is_synced = (ts.get("dataset_sha256") == current_sha)
                print(f"   • Último treino LoRA realizado em       : {ts.get('timestamp')} ({ts.get('samples_count')} exemplos)")
                print(f"   • Modelo LoRA sincronizado c/ dataset?  : {'✅ SIM (100% atualizado)' if is_synced else '⚠️  NÃO (há novos exemplos no dataset ainda não treinados no LoRA)'}")
            except Exception:
                pass
        else:
            print("   • Último treino LoRA realizado em       : ⚠️  Ainda não executado (output/train_state.json ausente)")
        print("=" * 75)
        return

    if not queue:
        logger.info("✨ Todos os documentos da base já estão aprendidos no dataset! Nenhum chunk novo pendente.")
        return

    logger.info(f"🚀 Iniciando síntese incremental de {len(queue)} chunks com o modelo '{args.model}'...")
    logger.info("💡 Salvamento automático ativado a cada chunk (você pode interromper com Ctrl+C sem perder dados).")

    new_examples_count = 0
    total_queue = len(queue)
    start_time = time.time()

    try:
        for idx, doc_chunk in enumerate(queue, start=1):
            doc_name = doc_chunk["name"]
            chunk_id = doc_chunk["chunk_id"]
            page = doc_chunk.get("page", 1)
            file_hash = doc_chunk.get("file_hash", "")

            elapsed = time.time() - start_time
            avg_time = elapsed / (idx - 1) if idx > 1 else 0
            eta_min = ((total_queue - idx + 1) * avg_time) / 60.0 if avg_time > 0 else 0
            eta_str = f" | ETA: {eta_min:.1f} min" if idx > 1 else ""

            logger.info(
                f"[{idx}/{total_queue}] Processando: {doc_name} (Pág. {page}){eta_str}..."
            )

            cases = generate_cases_from_text(
                context_text=doc_chunk["text"],
                doc_name=doc_name,
                model=args.model,
                count=args.cases_per_chunk
            )

            chunk_samples = []
            for c in cases:
                u_query = str(c.get("user_query", "")).strip()
                a_resp = str(c.get("assistant_response", "")).strip()
                if u_query and a_resp and u_query not in seen_queries:
                    chatml_sample = format_to_chatml(
                        system_prompt=SYSTEM_PROMPT,
                        user_query=u_query,
                        assistant_response=a_resp
                    )
                    chunk_samples.append(chatml_sample)
                    existing_samples.append(chatml_sample)
                    seen_queries.add(u_query)

            # Grava incrementalmente no .jsonl e no dataset_state.json imediatamente
            if chunk_samples:
                append_samples_to_jsonl(output_file, chunk_samples)
                new_examples_count += len(chunk_samples)

            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            state["processed_chunk_ids"][chunk_id] = {
                "document": doc_name,
                "page": page,
                "samples": len(chunk_samples),
                "timestamp": now_str
            }

            doc_entry = state["documents"].get(doc_name, {
                "file_hash": file_hash,
                "chunks_processed": 0,
                "samples_generated": 0,
                "updated_at": now_str
            })
            # Se o file_hash mudou, reinicia contagem daquele doc
            if file_hash and doc_entry.get("file_hash") and doc_entry["file_hash"] != file_hash:
                doc_entry["chunks_processed"] = 0
                doc_entry["samples_generated"] = 0

            doc_entry["file_hash"] = file_hash or doc_entry.get("file_hash", "")
            doc_entry["chunks_processed"] = doc_entry.get("chunks_processed", 0) + 1
            doc_entry["samples_generated"] = doc_entry.get("samples_generated", 0) + len(chunk_samples)
            doc_entry["updated_at"] = now_str
            state["documents"][doc_name] = doc_entry

            save_dataset_state(state_file, state)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupção detectada (Ctrl+C)! Todo o progresso até o último chunk já está salvo.")

    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Novos exemplos adicionados nesta execução : {new_examples_count}")
    logger.info(f"📚 Total acumulado de exemplos no dataset    : {len(existing_samples)}")
    logger.info(f"📄 Documentos aprendidos no estado           : {len(state.get('documents', {}))}")
    logger.info(f"💾 Arquivo de dataset                        : {output_file}")
    logger.info(f"🗂️  Arquivo de estado incremental            : {state_file}")
    logger.info("=" * 60)
    logger.info("✅ Dataset pronto para o treinamento!")


if __name__ == "__main__":
    main()
