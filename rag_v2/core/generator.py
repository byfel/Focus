import re
import ast
import json
import requests
from typing import List, Dict
from .config import OLLAMA_CHAT_URL, DEFAULT_LLM_MODEL


def is_code_or_script_request(query: str) -> bool:
    """
    Identifica se a consulta do usuário solicita criação, desenvolvimento
    ou auxílio na escrita de scripts, automações, códigos ou comandos.
    """
    q_lower = query.lower()
    code_indicators = [
        "script", "scripts", "código", "codigo", "code", "programa", "programar",
        "crie", "criar", "cria", "faça", "fazer", "escreva", "escrever", "gere", "gerar",
        "desenvolva", "desenvolver", "implemente", "implementar", "monte", "montar",
        "python", "bash", "shell", "sh", "powershell", "ps1", "batch", "bat",
        "ansible", "playbook", "dockerfile", "docker-compose", "yaml", "yml", "json",
        "regex", "sql", "automação", "automacao", "automatizar", "automatize",
        "função", "funcao", "function", "classe", "api", "curl", "webhook", "cron", "crontab"
    ]
    pattern = r'\b(' + '|'.join(re.escape(w) for w in code_indicators) + r')\b'
    return bool(re.search(pattern, q_lower))


def build_context(results: List[Dict], max_chunks: int = None, max_chars_per_chunk: int = None) -> str:
    if not results:
        return ""

    selected = results[:max_chunks] if max_chunks else results
    parts = []
    for i, item in enumerate(selected, 1):
        chunk = item.get("chunk", {})
        text = chunk.get("text", "")
        if max_chars_per_chunk and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        parts.append(
            f"""--- FONTE {i} ({chunk.get('document', 'desconhecido')}, Pág. {chunk.get('page', '?')}) ---
{text}
"""
        )

    return "\n".join(parts)


SOLVER_SYSTEM_PROMPT = """Você é o FOCUS AI - Solucionador Sênior e Especialista em Engenharia de Mídia, Broadcast, VFX, Áudio/Vídeo Profissional, Infraestrutura e Automação.

REGRA DE OURO DE QUALIDADE E PROFUNDIDADE TÉCNICA:
Sua resposta deve ser TÃO COMPLETA, DETALHADA E PRECISA quanto um guia de engenharia sênior, combinando todas as Fontes relevantes do CONTEXTO sem omitir nenhum detalhe técnico.

Siga rigorosamente estas diretrizes:
1. FIDELIDADE ABSOLUTA E RIQUEZA DE DETALHES (NÃO RESUMA DEMAIS):
   - Extraia e preserve EXATAMENTE todos os caminhos de diretórios (ex: `C:\\Users\\<usuario>\\...`, `/opt/...`, `/usr/OFX/...`), números de versão exatos (ex: `<= 0.87.0`, `<= 0.74.0`), nomes de scripts (ex: `corrigir_nodes.bat`, `corrigir_nodes.py`), portas de rede, variáveis e menus da interface (ex: `Settings > Libraries`, `Engine > Restart`).
   - Se o contexto apresentar mais de uma opção de solução (ex: Opção A via Script Automatizado e Opção B Manual), documente AMBAS de forma completa e passo a passo.
   - Cite as fontes utilizadas (ex: Conforme Fonte 1, Fonte 2) para dar rastreabilidade.
   - NUNCA misture assuntos de softwares diferentes que não tenham relação com a pergunta do usuário.

2. ESTRUTURA CLARA DE TROUBLESHOOTING E RESOLUÇÃO:
   Organize sua resposta em Markdown limpo e bem formatado contendo:
   - **Problema / Sintoma Identificado:** Descrição clara do erro ou cenário (incluindo mensagens de erro exatas na tela/log).
   - **Causa Raiz Documentada:** Por que o problema ocorre segundo a documentação.
   - **Solução Passo a Passo Detalhada:** Todos os passos numerados com caminhos completos, versões exatas, opções (A/B) e comandos.
   - **Explicação de Scripts / Automações (se houver):** O que cada script (`.bat`, `.py`, `.sh`) faz nos bastidores.
   - **Como Confirmar / Considerações Adicionais:** Checklist de validação final, backups necessários e cuidados com versões.

3. CRIAÇÃO DE SCRIPTS E CÓDIGOS:
   - Quando solicitado um script ou automação, forneça o código completo, pronto para produção e comentado em blocos Markdown (```bash, ```python, ```powershell, ```bat).

Responda sempre em português técnico, claro, completo e pronto para uso em produção."""


def format_solver_output_to_markdown(raw_text: str) -> str:
    """
    Se o modelo fine-tuned retornar um dicionário Python/JSON serializado
    (ex: {'SINTOMA': '...', 'FATOS_DOCUMENTADOS': [...]}), converte automaticamente
    para Markdown limpo e formatado para exibição na interface Web.
    """
    text = (raw_text or "").strip()
    if not text.startswith("{"):
        return text

    # Pega apenas o primeiro bloco de dicionário caso o modelo tenha repetido um segundo bloco
    first_dict_str = text.split("\n\n{")[0].strip()
    parsed = None
    for candidate in (first_dict_str, text):
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                break
            except Exception:
                continue

    if not isinstance(parsed, dict):
        return text

    md_parts = []
    for key, val in parsed.items():
        if val is None or val == [] or val == "":
            continue
        md_parts.append(f"### {key}")
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    for sub_k, sub_v in item.items():
                        lang = sub_k.lower() if sub_k.lower() in ("bash", "python", "sh", "powershell") else "bash"
                        md_parts.append(f"**{sub_k}:**\n```{lang}\n{sub_v}\n```")
                else:
                    md_parts.append(f"- {item}")
        elif isinstance(val, dict):
            for sub_k, sub_v in val.items():
                md_parts.append(f"- **{sub_k}:** {sub_v}")
        else:
            md_parts.append(str(val))
        md_parts.append("")

    return "\n".join(md_parts).strip()


def generate_answer(
    query: str,
    results: List[Dict],
    model: str = DEFAULT_LLM_MODEL
) -> str:
    is_code_request = is_code_or_script_request(query)
    is_solver_model = "focus-media-solver" in (model or "").lower()

    # Se não houver contexto documental recuperado:
    # 1. Se for modelo fine-tuned (focus-media-solver) ou pedido de código: responde com o conhecimento treinado
    # 2. Se for pergunta estrita sobre especificação em modelo base sem contexto: informa não encontrado
    if not results and not is_code_request and not is_solver_model:
        return "Essa informação não foi encontrada na documentação fornecida."

    # Para o modelo Solucionador (focus-media-solver), envia o contexto completo das fontes
    # e usa o SOLVER_SYSTEM_PROMPT detalhado para garantir profundidade técnica igual/superior ao Gemma
    if is_solver_model:
        full_ctx = build_context(results) if results else "Nenhum manual específico encontrado na base local."
        user_msg = f"""CONTEXTO DA BASE TÉCNICA:

{full_ctx}

SOLICITAÇÃO DO USUÁRIO: {query}

Elabore um guia técnico completo e detalhado combinando todas as informações relevantes das fontes acima (preservando todos os caminhos, versões exatas, scripts e passos):"""
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 8192,
                        "num_predict": 2048,
                        "repeat_penalty": 1.12,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            raw_ans = response.json().get("message", {}).get("content", "").strip()
            return format_solver_output_to_markdown(raw_ans)
        except Exception as e:
            print(f"❌ Erro na geração (solver): {e}")
            return f"Erro ao gerar resposta: {str(e)}"

    context = (
        build_context(results)
        if results
        else "Nenhum manual específico encontrado na base local para este tópico. Utilize suas melhores práticas de engenharia e programação."
    )

    system_prompt = """Você é um assistente técnico sênior de infraestrutura, engenharia de broadcast e desenvolvimento/automação.

Você opera sob duas diretrizes fundamentais:

1. FIDELIDADE E PRECISÃO AOS MANUAIS (Para Consultas sobre Documentação):
- Quando a pergunta for sobre procedimentos oficiais, parâmetros, portas de rede, caminhos de instalação ou especificações de softwares/equipamentos documentados no CONTEXTO:
  - Seja rigorosamente fiel aos manuais presentes no CONTEXTO.
  - Preserve exatamente comandos, sintaxe de terminal, opções, portas de rede, nomes de arquivos e parâmetros de configuração oficiais.
  - Se o contexto estiver em inglês, traduza as explicações e procedimentos técnicos para português claro.
  - Para sistemas Linux: reconheça que Rocky Linux e AlmaLinux são equivalentes a RHEL/CentOS. Aplique os comandos correspondentes com essa equivalência.
  - Se a pergunta for estritamente conceitual ou documental sobre algo que NÃO consta no contexto e NÃO for um pedido de script/código, responda: "Essa informação não foi encontrada na documentação fornecida."

2. LIBERDADE PARA CRIAÇÃO DE SCRIPTS & CÓDIGOS (Engenharia e Automação):
- Quando o usuário solicitar a CRIAÇÃO, GERAÇÃO, ADAPTAÇÃO ou MELHORIA de SCRIPTS, CÓDIGOS ou AUTOMAÇÕES (Bash, Python, PowerShell, Ansible, Docker, regex, crontab, etc.):
  - Você está TOTALMENTE LIBERADO para usar todo o seu conhecimento geral de programação, boas práticas de engenharia e administração de sistemas.
  - Se o CONTEXTO contiver parâmetros técnicos relevantes (portas oficiais, caminhos de diretórios, comandos do software abordado), INCORPORE-OS com precisão dentro do código/script.
  - Se o contexto não tiver o script pronto, DESENVOLVA o código completo, funcional, modular, comentado e com tratamento de erros.
  - Apresente o código pronto para uso em blocos markdown com a linguagem correspondente (ex: ```bash, ```python, ```powershell).
  - Inclua instruções breves de execução, pré-requisitos (permissões, dependências) e explique o funcionamento do script.

DIRETRIZ GERAL:
Responda em português, de maneira técnica, profissional, clara e pronta para uso em produção."""

    user_prompt = f"""CONTEXTO DA BASE TÉCNICA:

{context}

SOLICITAÇÃO DO USUÁRIO: {query}

RESPOSTA:"""

    temperature = 0.2 if is_code_request else 0.1

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=600
        )
        response.raise_for_status()

        return response.json().get("message", {}).get("content", "").strip()

    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return f"Erro ao gerar resposta: {str(e)}"
