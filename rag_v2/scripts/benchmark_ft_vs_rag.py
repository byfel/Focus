#!/usr/bin/env python3
"""
FOCUS AI - Benchmark Comparativo de Velocidade e Qualidade Técnica:
  1. Fine-Tuning Direto (focus-media-solver puro via Ollama, sem overhead de busca)
  2. Híbrido: Fine-Tuning + RAG V2 (focus-media-solver + Qdrant nomic-embed-text)
  3. RAG Tradicional (gemma3:12b + Qdrant nomic-embed-text)

Avalia:
  - Latência (segundos) e Vazão (chars/s)
  - Aderência ao Protocolo Técnico (5 seções de Troubleshooting)
  - Precisão Factual (Fact-Check automático de portas, caminhos, comandos e conceitos reais)
  - Exibição das respostas completas para inspeção humana

Uso:
    python3 /opt/ai/rag_v2/scripts/benchmark_ft_vs_rag.py --limit 3 --show-answers
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

PROJECT_V2_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_V2_ROOT))

from core.config import OLLAMA_CHAT_URL
from core.pipeline import ask as ask_rag_v2

# ==============================================================================
# CASOS DE TESTE COM GABARITO TÉCNICO (GROUND-TRUTH FACTS)
# ==============================================================================

BENCHMARK_CASES = [
    {
        "question": "No Dante Controller, o nome de um conversor de áudio está em vermelho e não roteia sinal. O que significa e como corrigir?",
        "expected_facts": ["sub-rede", "ip", "dhcp", "device view", "network config"],
        "fact_aliases": {
            "sub-rede": ["sub-rede", "subrede", "subnet", "faixa de ip", "máscara"],
            "ip": ["ip", "169.254", "192.168", "endereço"],
            "dhcp": ["dhcp", "automatically", "estático", "automático"],
            "device view": ["device view", "duplo clique", "configuração do dispositivo"],
            "network config": ["network config", "aba configure", "rede"],
        },
    },
    {
        "question": "Instalei o Boris FX Sapphire no Rocky Linux para o Autodesk Flame, mas os nós OFX não aparecem no Batch. Como diagnosticar?",
        "expected_facts": ["/usr/ofx/plugins", "sapphire.ofx.bundle", "ofxplugincache", "symlink", "/opt/borisfx"],
        "fact_aliases": {
            "/usr/ofx/plugins": ["/usr/ofx/plugins", "ofx/plugins"],
            "sapphire.ofx.bundle": ["sapphire.ofx.bundle", ".bundle", "sapphireofx"],
            "ofxplugincache": ["ofxplugincache", "cache", "status"],
            "symlink": ["ln -s", "symlink", "link simbólico", "ls -la"],
            "/opt/borisfx": ["/opt/borisfx", "rpm -ql"],
        },
    },
    {
        "question": "Clientes PCoIP (HP Anyware) estão sofrendo desconexões constantes ao acessar as ilhas Linux. Quais portas e logs verificar?",
        "expected_facts": ["4172", "udp", "/var/log/pcoip", "mtu", "firewall"],
        "fact_aliases": {
            "4172": ["4172"],
            "udp": ["udp"],
            "/var/log/pcoip": ["/var/log/pcoip", "pcoip-agent"],
            "mtu": ["mtu", "1300", "1400", "fragmentação"],
            "firewall": ["firewall", "iptables", "ss -tulnp", "netstat"],
        },
    },
    {
        "question": "Como verificar e reiniciar os serviços do Autodesk Wiretap e Stone+Wire no terminal Linux?",
        "expected_facts": ["adsk_wiretap", "adsk_stone_wire", "7100", "systemctl"],
        "fact_aliases": {
            "adsk_wiretap": ["adsk_wiretap", "wiretap"],
            "adsk_stone_wire": ["adsk_stone_wire", "stone+wire", "stone_wire"],
            "7100": ["7100", "7104"],
            "systemctl": ["systemctl", "service", "restart"],
        },
    },
    {
        "question": "O que é uma Timeline no Avid Media Composer e como organizar faixas de vídeo e áudio?",
        "expected_facts": ["sequência", "faixas", "vídeo", "áudio", "timecode"],
        "fact_aliases": {
            "sequência": ["sequência", "sequence", "edição", "cortes"],
            "faixas": ["faixas", "tracks", "v1", "a1"],
            "vídeo": ["vídeo", "video"],
            "áudio": ["áudio", "audio"],
            "timecode": ["timecode", "tempo", "sincronismo", "playhead"],
        },
    },
    {
        "question": "Quais são as boas práticas de configuração de rede para tráfego de áudio Dante sobre IP?",
        "expected_facts": ["qos", "dscp", "igmp", "gigabit", "eee"],
        "fact_aliases": {
            "qos": ["qos", "qualidade de serviço", "prioridade"],
            "dscp": ["dscp", "diffserv", "ptp", "clock"],
            "igmp": ["igmp", "multicast", "snooping"],
            "gigabit": ["gigabit", "1gbps", "switch"],
            "eee": ["eee", "energy efficient", "green ethernet", "vlan"],
        },
    },
    {
        "question": "Como diagnosticar lentidão de leitura de mídia em armazenamento compartilhado nas estações de edição Linux?",
        "expected_facts": ["iostat", "df", "mount", "rede", "throughput"],
        "fact_aliases": {
            "iostat": ["iostat", "iotop", "dd", "fio", "latência"],
            "df": ["df -h", "espaço", "capacidade"],
            "mount": ["mount", "nfs", "smb", "san", "nas", "stone+wire"],
            "rede": ["ethtool", "iperf", "interface", "10g", "fibra"],
            "throughput": ["throughput", "mb/s", "leitura", "escrita", "disco"],
        },
    },
    {
        "question": "Crie um script rápido em Bash para testar se a porta TCP 7100 do Wiretap e UDP 4172 do PCoIP estão ativas.",
        "expected_facts": ["#!/", "7100", "4172", "ss", "echo"],
        "fact_aliases": {
            "#!/": ["#!/bin/bash", "#!/usr/bin/env bash"],
            "7100": ["7100"],
            "4172": ["4172"],
            "ss": ["ss ", "netstat", "nc ", "lsof"],
            "echo": ["echo", "log"],
        },
    },
]

STRUCTURED_SECTIONS = [
    "SINTOMA",
    "FATOS",
    "CAUSA",
    "COMO_CONFIRMAR",
    "ACAO_RECOMENDADA",
]


def warmup_ollama_model(model_name: str):
    """Carrega o modelo na VRAM antes do cronômetro iniciar para evitar penalidade de cold-start."""
    print(f"🔥 Aquecendo modelo '{model_name}' na GPU (warmup)...", end=" ", flush=True)
    t0 = time.time()
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Responda apenas: OK"}],
                "stream": False,
                "options": {"num_predict": 5, "temperature": 0.0},
            },
            timeout=180,
        )
        resp.raise_for_status()
        print(f"pronto em {time.time() - t0:.2f}s!")
    except Exception as e:
        print(f"aviso no warmup ({e})")


def ask_finetuned_direct(question: str, model_name: str) -> Dict:
    """Consulta diretamente o modelo Fine-Tuned no Ollama (sem busca Qdrant)."""
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 768, "repeat_penalty": 1.15},
        },
        timeout=180,
    )
    resp.raise_for_status()
    answer = resp.json().get("message", {}).get("content", "").strip()
    return {"answer": answer, "sources": []}


def check_protocol_score(answer: str) -> int:
    """Conta quantas seções estruturadas de Troubleshooting aparecem na resposta (0 a 5)."""
    upper = answer.upper()
    return sum(1 for sec in STRUCTURED_SECTIONS if sec in upper)


def evaluate_technical_accuracy(answer: str, case: Dict) -> Dict:
    """Verifica quantos fatos técnicos do gabarito estão presentes na resposta."""
    lower_ans = answer.lower()
    aliases_map = case.get("fact_aliases", {})
    matched = []
    missing = []

    for fact_key in case.get("expected_facts", []):
        synonyms = aliases_map.get(fact_key, [fact_key])
        if any(s.lower() in lower_ans for s in synonyms):
            matched.append(fact_key)
        else:
            missing.append(fact_key)

    total = len(case.get("expected_facts", [])) or 1
    accuracy_pct = round((len(matched) / total) * 100, 1)
    return {
        "accuracy_pct": accuracy_pct,
        "matched_facts": matched,
        "missing_facts": missing,
    }


def run_benchmark_mode(
    mode_title: str,
    model_name: str,
    runner_fn: Callable[[str], Dict],
    cases: List[Dict],
) -> Dict:
    print(f"\n{'=' * 85}")
    print(f"🚀 MODO: {mode_title}")
    print(f"{'=' * 85}")

    warmup_ollama_model(model_name)

    items = []
    total_time = 0.0
    total_chars = 0
    total_protocol_hits = 0
    total_accuracy_pct = 0.0

    for idx, case in enumerate(cases, start=1):
        q = case["question"]
        short_q = q[:68] + "..." if len(q) > 68 else q
        print(f"\n  [{idx}/{len(cases)}] {short_q}")
        t0 = time.time()
        try:
            res = runner_fn(q)
            elapsed = time.time() - t0
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            chars = len(answer)
            chars_per_sec = chars / elapsed if elapsed > 0 else 0.0
            proto_score = check_protocol_score(answer)
            fact_eval = evaluate_technical_accuracy(answer, case)
            acc_pct = fact_eval["accuracy_pct"]

            total_time += elapsed
            total_chars += chars
            total_protocol_hits += proto_score
            total_accuracy_pct += acc_pct

            print(
                f"     ⏱️  {elapsed:6.2f}s | ⚡ {chars_per_sec:5.1f} ch/s | "
                f"🎯 Estrutura: {proto_score}/5 | 🧠 Precisão Técnica: {acc_pct:5.1f}% "
                f"({len(fact_eval['matched_facts'])}/{len(case['expected_facts'])} fatos: {', '.join(fact_eval['matched_facts'])})"
            )

            items.append({
                "question": q,
                "time_sec": round(elapsed, 2),
                "chars": chars,
                "chars_per_sec": round(chars_per_sec, 1),
                "sources_count": len(sources),
                "protocol_score": proto_score,
                "accuracy_pct": acc_pct,
                "matched_facts": fact_eval["matched_facts"],
                "missing_facts": fact_eval["missing_facts"],
                "full_answer": answer,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"     ❌ Erro após {elapsed:.2f}s: {e}")
            items.append({
                "question": q,
                "time_sec": round(elapsed, 2),
                "accuracy_pct": 0.0,
                "protocol_score": 0,
                "error": str(e),
                "full_answer": "",
            })

    count = len(cases) or 1
    avg_time = total_time / count
    avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0.0
    avg_proto = total_protocol_hits / count
    avg_acc = total_accuracy_pct / count

    print(f"\n  📊 Resumo — {mode_title}:")
    print(f"     • Tempo Médio por Resposta : {avg_time:.2f}s")
    print(f"     • Vazão Média de Geração   : {avg_chars_per_sec:.1f} chars/s")
    print(f"     • Aderência ao Protocolo   : {avg_proto:.1f}/5 seções técnicas")
    print(f"     • Precisão Técnica Factual : {avg_acc:.1f}% dos fatos-chave")

    return {
        "title": mode_title,
        "model": model_name,
        "avg_time_sec": round(avg_time, 2),
        "total_time_sec": round(total_time, 2),
        "avg_chars_per_sec": round(avg_chars_per_sec, 1),
        "avg_protocol_score": round(avg_proto, 2),
        "avg_accuracy_pct": round(avg_acc, 1),
        "results": items,
    }


def print_answers_inspection(
    ft_direct: Dict,
    ft_rag: Dict,
    base_rag: Dict,
    cases: List[Dict],
):
    print("\n" + "=" * 95)
    print("🔍 INSPEÇÃO DE QUALIDADE DAS RESPOSTAS (LADO A LADO)")
    print("=" * 95)

    for i, case in enumerate(cases):
        q = case["question"]
        r_ft = ft_direct["results"][i]
        r_hyb = ft_rag["results"][i]
        r_gem = base_rag["results"][i]

        print(f"\n📌 PERGUNTA {i + 1}: {q}")
        print(f"   Gabarito esperado: {', '.join(case['expected_facts'])}")
        print("-" * 95)

        print(
            f"🟢 [1] FINE-TUNING PURO ({ft_direct['model']}) — "
            f"{r_ft.get('time_sec')}s | Precisão: {r_ft.get('accuracy_pct')}% | Estrutura: {r_ft.get('protocol_score')}/5:"
        )
        print(r_ft.get("full_answer", "(vazio)"))
        print("-" * 95)

        print(
            f"🔵 [2] HÍBRIDO FT + RAG ({ft_rag['model']} + Qdrant) — "
            f"{r_hyb.get('time_sec')}s | Precisão: {r_hyb.get('accuracy_pct')}% | Estrutura: {r_hyb.get('protocol_score')}/5:"
        )
        print(r_hyb.get("full_answer", "(vazio)"))
        print("-" * 95)

        print(
            f"🟠 [3] RAG TRADICIONAL ({base_rag['model']} + Qdrant) — "
            f"{r_gem.get('time_sec')}s | Precisão: {r_gem.get('accuracy_pct')}% | Estrutura: {r_gem.get('protocol_score')}/5:"
        )
        gem_ans = r_gem.get("full_answer", "(vazio)")
        print(gem_ans[:1200] + ("\n... [truncado para exibição]" if len(gem_ans) > 1200 else ""))
        print("=" * 95)


def print_comparison_table(
    ft_direct: Dict,
    ft_rag: Dict,
    base_rag: Dict,
    cases: List[Dict],
):
    print("\n" + "=" * 95)
    print("🏆 RELATÓRIO FINAL: VELOCIDADE & QUALIDADE TÉCNICA")
    print("=" * 95)

    header = (
        f"{'Pergunta':<38} | {'FT Direto':>10} | {'FT + RAG':>10} | "
        f"{'Gemma3 RAG':>10} | {'Speedup FT':>11}"
    )
    print("\n" + header)
    print("-" * 95)

    for i, case in enumerate(cases):
        q = case["question"]
        t_ft = ft_direct["results"][i].get("time_sec", 0.0)
        t_hyb = ft_rag["results"][i].get("time_sec", 0.0)
        t_gem = base_rag["results"][i].get("time_sec", 0.0)

        speedup = (t_gem / t_ft) if t_ft > 0 else 0.0
        label = q[:36] + ".." if len(q) > 36 else q
        print(
            f"{label:<38} | {t_ft:>9.2f}s | {t_hyb:>9.2f}s | "
            f"{t_gem:>9.2f}s | {speedup:>9.2f}x 🚀"
        )

    print("-" * 95)
    avg_ft = ft_direct["avg_time_sec"]
    avg_hyb = ft_rag["avg_time_sec"]
    avg_gem = base_rag["avg_time_sec"]
    overall_speedup_direct = (avg_gem / avg_ft) if avg_ft > 0 else 0.0
    overall_speedup_hybrid = (avg_gem / avg_hyb) if avg_hyb > 0 else 0.0

    print(
        f"{'TEMPO MÉDIO POR PERGUNTA':<38} | {avg_ft:>9.2f}s | {avg_hyb:>9.2f}s | "
        f"{avg_gem:>9.2f}s | {overall_speedup_direct:>9.2f}x ⚡"
    )
    print(
        f"{'PRECISÃO TÉCNICA FACTUAL (%)':<38} | "
        f"{ft_direct['avg_accuracy_pct']:>8.1f}% | {ft_rag['avg_accuracy_pct']:>8.1f}% | "
        f"{base_rag['avg_accuracy_pct']:>8.1f}% |"
    )
    print(
        f"{'ESTRUTURA TROUBLESHOOTING (0 a 5)':<38} | "
        f"{ft_direct['avg_protocol_score']:>7.1f}/5 | {ft_rag['avg_protocol_score']:>7.1f}/5 | "
        f"{base_rag['avg_protocol_score']:>7.1f}/5 |"
    )
    print(
        f"{'VAZÃO MÉDIA (CHARS/SEGUNDO)':<38} | "
        f"{ft_direct['avg_chars_per_sec']:>8.1f}/s | {ft_rag['avg_chars_per_sec']:>8.1f}/s | "
        f"{base_rag['avg_chars_per_sec']:>8.1f}/s |"
    )
    print("=" * 95)

    print("\n📌 CONCLUSÃO EXECUTIVA:")
    if avg_ft > 0 and avg_gem > 0:
        pct_faster_direct = ((avg_gem - avg_ft) / avg_gem) * 100
        print(
            f"  1. Fine-Tuning Puro ({ft_direct['model']}): {overall_speedup_direct:.2f}x mais rápido "
            f"({pct_faster_direct:.1f}% menos espera) | Precisão Técnica: {ft_direct['avg_accuracy_pct']:.1f}% | "
            f"Estrutura: {ft_direct['avg_protocol_score']:.1f}/5."
        )
    if avg_hyb > 0 and avg_gem > 0:
        pct_faster_hybrid = ((avg_gem - avg_hyb) / avg_gem) * 100
        print(
            f"  2. Fine-Tuning + RAG Híbrido ({ft_rag['model']} + Qdrant): {overall_speedup_hybrid:.2f}x mais rápido "
            f"({pct_faster_hybrid:.1f}% menos espera) | Precisão Técnica: {ft_rag['avg_accuracy_pct']:.1f}% | "
            f"Estrutura: {ft_rag['avg_protocol_score']:.1f}/5."
        )
    print(
        f"  3. RAG Tradicional ({base_rag['model']} + Qdrant): {avg_gem:.2f}s por resposta | "
        f"Precisão Técnica: {base_rag['avg_accuracy_pct']:.1f}% | Estrutura: {base_rag['avg_protocol_score']:.1f}/5."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Fine-Tuning (focus-media-solver) vs RAG (gemma3:12b)"
    )
    parser.add_argument("--ft-model", type=str, default="focus-media-solver",
                        help="Nome do modelo Fine-Tuned no Ollama (padrão: focus-media-solver)")
    parser.add_argument("--rag-model", type=str, default="gemma3:12b",
                        help="Nome do modelo base para comparação no RAG (padrão: gemma3:12b)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limitar quantidade de perguntas do benchmark (0 = todas as 8)")
    parser.add_argument("--show-answers", action="store_true", default=True,
                        help="Exibir as respostas completas geradas por cada modo (padrão: ativado)")
    parser.add_argument("--no-show-answers", action="store_false", dest="show_answers",
                        help="Ocultar o texto completo das respostas no terminal")
    args = parser.parse_args()

    cases = BENCHMARK_CASES[:args.limit] if args.limit > 0 else BENCHMARK_CASES

    print("=" * 95)
    print("⚡ BENCHMARK FOCUS AI: VELOCIDADE & QUALIDADE TÉCNICA (FT vs RAG)")
    print(f"   Data/Hora : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Perguntas : {len(cases)}")
    print("=" * 95)

    # 1. Fine-Tuning Direto (Sem busca no Qdrant)
    ft_direct_stats = run_benchmark_mode(
        mode_title=f"1. FINE-TUNING PURO ({args.ft_model} direto, sem busca RAG)",
        model_name=args.ft_model,
        runner_fn=lambda q: ask_finetuned_direct(q, args.ft_model),
        cases=cases,
    )

    # 2. Híbrido: Fine-Tuning + RAG V2 (Qdrant + focus-media-solver)
    ft_rag_stats = run_benchmark_mode(
        mode_title=f"2. HÍBRIDO: FINE-TUNING + RAG V2 ({args.ft_model} + Qdrant)",
        model_name=args.ft_model,
        runner_fn=lambda q: ask_rag_v2(q, model=args.ft_model),
        cases=cases,
    )

    # 3. RAG Tradicional (Qdrant + gemma3:12b)
    base_rag_stats = run_benchmark_mode(
        mode_title=f"3. RAG TRADICIONAL ({args.rag_model} + Qdrant)",
        model_name=args.rag_model,
        runner_fn=lambda q: ask_rag_v2(q, model=args.rag_model),
        cases=cases,
    )

    if args.show_answers:
        print_answers_inspection(ft_direct_stats, ft_rag_stats, base_rag_stats, cases)

    print_comparison_table(ft_direct_stats, ft_rag_stats, base_rag_stats, cases)

    # Salva relatório JSON completo (incluindo o texto integral de todas as respostas)
    out_dir = PROJECT_V2_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_ft_vs_rag_{ts}.json"

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions_count": len(cases),
        "ft_direct": ft_direct_stats,
        "ft_rag_hybrid": ft_rag_stats,
        "rag_baseline": base_rag_stats,
    }
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Relatório completo com todas as respostas salvo em: {out_file}\n")


if __name__ == "__main__":
    main()
