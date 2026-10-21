import re
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


SOLVER_SYSTEM_PROMPT = """Você é o FOCUS AI - Solucionador Sênior e Especialista em Engenharia de Mídia, Broadcast, Áudio/Vídeo Profissional, Infraestrutura e Automação de Estúdios.

Sua metodologia de resolução segue o protocolo estruturado de Troubleshooting Técnico:
1. SINTOMA: Descrição precisa do problema relatado.
2. FATOS_DOCUMENTADOS: Fatos comprovados e extraídos diretamente das evidências técnicas.
3. CAUSAS_DOCUMENTADAS / CAUSAS_DEDUZIVEIS: Relação técnica causal sem alucinar.
4. HIPOTESES: Possibilidades técnicas plausíveis quando a evidência exigir confirmação.
5. COMO_CONFIRMAR: Procedimentos práticos no terminal (Bash/Linux/PowerShell) ou interface (GUI) para validar a causa raiz.
6. ACAO_RECOMENDADA: Passo a passo claro com comandos exatos, scripts, parâmetros e caminhos de arquivo.
7. LIMITACOES: O que a evidência não permitiu aferir com certeza.

Ao criar scripts ou códigos (Bash, Python, automações):
- Forneça código completo, pronto para produção, modular, com validação de erros e paths reais.
- Preserve equivalências de sistemas (ex: Rocky Linux / AlmaLinux / CentOS / RHEL).
Responda sempre em português técnico, objetivo e com alto rigor de engenharia."""


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

    # Para o modelo Fine-Tuned (focus-media-solver), preserva o System Prompt e formato de treino,
    # enviando um resumo enxuto das Top-2 referências para não saturar a janela de contexto de 512 tokens
    if is_solver_model:
        short_ctx = build_context(results, max_chunks=2, max_chars_per_chunk=350) if results else ""
        user_msg = f"{query}\n\n[Referência rápida da base:\n{short_ctx}]" if short_ctx else query
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
                        "num_predict": 768,
                        "repeat_penalty": 1.15,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
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
