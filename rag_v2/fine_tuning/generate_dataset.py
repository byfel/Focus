#!/usr/bin/env python3
"""
FOCUS AI - Sintetizador de Dataset para Fine-Tuning de Solucionador de Mídia.

Transforma manuais técnicos, KBs (Markdown/PDFs) e chunks do Qdrant em pares
instrucionais de treinamento (ChatML / JSONL) estruturados sob a metodologia rigorosa
de troubleshooting de broadcast e pós-produção (TSHOOT / Root Cause Analysis).

Uso:
    # 1. Gerar dataset a partir dos documentos na pasta data/documents + seeds de mídia:
    python3 generate_dataset.py --source files --include-seeds

    # 2. Gerar dataset a partir dos chunks indexados no Qdrant:
    python3 generate_dataset.py --source qdrant --max-samples 100

    # 3. Gerar apenas o dataset semente (Gold Standard) imediatamente (sem chamar LLM):
    python3 generate_dataset.py --source seeds-only
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

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

Abaixo está um trecho de DOCUMENTAÇÃO TÉCNICA / MANUAL / KB:
----------------------------------------
{context}
----------------------------------------

Sua tarefa:
Gere {count} caso(s) de suporte/troubleshooting realista(s) baseados estritamente ou inspirados nas tecnologias deste texto.
Cada caso deve conter:
1. Pergunta / Incidente do Usuário (Sintoma relatado por um operador, montador de Flame/Avid/Resolve, ou engenheiro de broadcast).
2. Resposta do Solucionador de Mídia, rigorosamente estruturada no formato:

SINTOMA:
<descrição do problema>

FATOS_DOCUMENTADOS:
- <itens extraídos do contexto>

CAUSAS_DOCUMENTADAS:
- <causas explicadas pelo texto>

COMO_CONFIRMAR:
1. <comandos de terminal (ex: ls -la, rpm, systemctl, netstat) ou inspeção na GUI>

ACAO_RECOMENDADA:
1. <comandos de correção, symlinks, permissões, limpeza de cache ou configuração>

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
# Casos reais de engenharia de broadcast e pós-produção para treino imediato
# ==============================================================================

GOLD_STANDARD_SEEDS = [
    {
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
# FUNÇÕES DE EXTRAÇÃO E SÍNTESE
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path, max_pages: int = 15) -> str:
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
    """Lê arquivos markdown, texto ou PDFs locais."""
    docs = []
    if not docs_dir.exists():
        logger.warning(f"Diretório de documentos não encontrado: {docs_dir}")
        return docs

    for file_path in docs_dir.glob("**/*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in [".md", ".txt"]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if len(content) > 100:
                    docs.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "text": content
                    })
            except Exception as e:
                logger.error(f"Erro ao ler {file_path}: {e}")
        elif ext == ".pdf":
            content = extract_text_from_pdf(file_path)
            if len(content) > 200:
                docs.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "text": content
                })
    return docs


def load_chunks_from_file(chunks_file: Path, limit: int = 200) -> List[Dict]:
    """Lê trechos já extraídos e limpos de chunks.json (do RAG V1/V2)."""
    if not chunks_file.exists():
        logger.warning(f"Arquivo chunks.json não encontrado em: {chunks_file}")
        return []
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = []
        items = data if isinstance(data, list) else list(data.values())
        logger.info(f"Carregando chunks de {chunks_file} (total no arquivo: {len(items)}, limite: {limit})...")
        for item in items[:limit]:
            text = (item.get("text") or item.get("content") or "").strip()
            if len(text) > 200:
                chunks.append({
                    "name": item.get("document", "chunk"),
                    "page": item.get("page", 1),
                    "text": text
                })
        return chunks
    except Exception as e:
        logger.error(f"Erro ao carregar chunks de {chunks_file}: {e}")
        return []


def load_chunks_from_qdrant(limit: int = 100) -> List[Dict]:
    """Scrolla chunks diretamente do Qdrant da base do FOCUS."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        records, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        chunks = []
        for r in records:
            p = r.payload or {}
            text = p.get("text", "").strip()
            if len(text) > 200:
                chunks.append({
                    "name": p.get("document", "qdrant_chunk"),
                    "page": p.get("page", 1),
                    "text": text
                })
        return chunks
    except Exception as e:
        logger.warning(f"Não foi possível conectar ao Qdrant ({QDRANT_URL}): {e}")
        return []


def generate_cases_from_text(
    context_text: str,
    doc_name: str,
    model: str = DEFAULT_LLM_MODEL,
    count: int = 2
) -> List[Dict]:
    """Envia o trecho do manual para o Ollama e gera casos de suporte estruturados."""
    prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(context=context_text[:3000], count=count)

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Você é um assistente que gera dados estritamente em JSON."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=120
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response}
        ]
    }


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="FOCUS AI - Gerador de Dataset para Fine-Tuning")
    parser.add_argument("--source", choices=["files", "chunks", "qdrant", "seeds-only", "all"], default="chunks",
                        help="Fonte: 'chunks' (chunks.json da V1/V2), 'files' (diretório de PDFs/MDs), 'qdrant', 'seeds-only' ou 'all'")
    parser.add_argument("--docs-dir", type=str, default=None,
                        help="Diretório de documentos/manuais (ex: /opt/ai/rag/documents)")
    parser.add_argument("--chunks-file", type=str, default=None,
                        help="Caminho para chunks.json (padrão: /opt/ai/rag/database/chunks.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Caminho do arquivo .jsonl gerado")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                        help="Modelo do Ollama para geração sintética")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Número máximo de trechos a processar com o LLM")
    parser.add_argument("--include-seeds", action="store_true", default=True,
                        help="Incluir exemplos sementes de alta qualidade (padrão: True)")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output) if args.output else output_dir / "media_solver_train.jsonl"

    dataset = []

    # 1. Adiciona os exemplos de ouro (Gold Standard Seeds)
    if args.include_seeds or args.source == "seeds-only":
        logger.info(f"Adicionando {len(GOLD_STANDARD_SEEDS)} casos sementes de alta fidelidade...")
        for seed in GOLD_STANDARD_SEEDS:
            chatml_sample = format_to_chatml(
                system_prompt=SYSTEM_PROMPT,
                user_query=seed["user_query"],
                assistant_response=seed["assistant_response"]
            )
            dataset.append(chatml_sample)

    if args.source == "seeds-only":
        logger.info(f"Salvando dataset com apenas sementes ({len(dataset)} exemplos)...")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"✅ Concluído! Dataset salvo em: {output_file}")
        return

    # 2. Carrega trechos de documentos
    docs_to_process = []

    # Opção A: chunks.json (Mais rápido e já processado)
    if args.source in ["chunks", "all"]:
        candidate_chunks = [
            Path(args.chunks_file) if args.chunks_file else None,
            Path(DATA_ROOT) / "database" / "chunks.json",
            Path(PROJECT_ROOT) / "data" / "database" / "chunks.json",
            Path("/opt/ai/rag/database/chunks.json"),
        ]
        chosen_chunks = next((p for p in candidate_chunks if p and p.exists()), None)
        if chosen_chunks:
            logger.info(f"Utilizando chunks processados de: {chosen_chunks}")
            docs_to_process.extend(load_chunks_from_file(chosen_chunks, limit=args.max_samples))
        else:
            logger.warning("Nenhum chunks.json encontrado nos caminhos padrão.")

    # Opção B: arquivos diretos (PDFs, Markdown, etc.)
    if args.source in ["files", "all"]:
        candidate_dirs = []
        if args.docs_dir:
            candidate_dirs.append(Path(args.docs_dir))
        else:
            candidate_dirs.extend([
                Path(DATA_ROOT) / "documents",
                Path(PROJECT_ROOT) / "data" / "documents",
                Path("/opt/ai/rag/documents"),
                Path(__file__).resolve().parent.parent / "data" / "documents",
            ])

        scanned_dirs = set()
        for d in candidate_dirs:
            if d.exists() and str(d.resolve()) not in scanned_dirs:
                scanned_dirs.add(str(d.resolve()))
                logger.info(f"Buscando manuais e documentos em: {d}")
                docs_to_process.extend(load_documents_from_dir(d))

    # Opção C: Qdrant
    if args.source in ["qdrant", "all"]:
        logger.info(f"Buscando chunks do Qdrant ({QDRANT_COLLECTION})...")
        docs_to_process.extend(load_chunks_from_qdrant(limit=args.max_samples))

    if not docs_to_process:
        logger.warning("Nenhum documento encontrado para sintetizar via LLM.")
    else:
        logger.info(f"Total de {len(docs_to_process)} documentos/chunks carregados. Gerando casos com {args.model}...")

        processed = 0
        for doc in docs_to_process[:args.max_samples]:
            processed += 1
            logger.info(f"[{processed}/{min(len(docs_to_process), args.max_samples)}] Processando: {doc['name']}...")
            cases = generate_cases_from_text(doc["text"], doc["name"], model=args.model, count=2)
            for c in cases:
                if "user_query" in c and "assistant_response" in c:
                    chatml_sample = format_to_chatml(
                        system_prompt=SYSTEM_PROMPT,
                        user_query=c["user_query"],
                        assistant_response=c["assistant_response"]
                    )
                    dataset.append(chatml_sample)

    logger.info(f"\n==================================================")
    logger.info(f"Total de exemplos gerados no dataset: {len(dataset)}")
    logger.info(f"Salvando em: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✅ Sucesso! Dataset pronto para o treinamento.")


if __name__ == "__main__":
    main()
