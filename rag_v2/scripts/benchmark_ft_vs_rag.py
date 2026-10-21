#!/usr/bin/env python3
"""
FOCUS AI - Benchmark Comparativo de Velocidade e Qualidade:
  1. Fine-Tuning Direto (focus-media-solver puro via Ollama, sem overhead de busca)
  2. Híbrido: Fine-Tuning + RAG V2 (focus-media-solver + Qdrant nomic-embed-text)
  3. RAG Tradicional (gemma3:12b + Qdrant nomic-embed-text)

Realiza warmup prévio de cada modelo na GPU para medir o tempo real de inferência,
compara latência (segundos), speedup (x vezes mais rápido), tamanho da resposta e
aderência ao protocolo estruturado de Troubleshooting do FOCUS.

Uso:
    python3 /opt/ai/rag_v2/scripts/benchmark_ft_vs_rag.py
    python3 /opt/ai/rag_v2/scripts/benchmark_ft_vs_rag.py --limit 5
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
# CASOS DE TESTE REAIS (MÍDIA, BROADCAST & INFRAESTRUTURA)
# ==============================================================================

BENCHMARK_QUESTIONS = [
    "No Dante Controller, o nome de um conversor de áudio está em vermelho e não roteia sinal. O que significa e como corrigir?",
    "Instalei o Boris FX Sapphire no Rocky Linux para o Autodesk Flame, mas os nós OFX não aparecem no Batch. Como diagnosticar?",
    "Clientes PCoIP (HP Anyware) estão sofrendo desconexões constantes ao acessar as ilhas Linux. Quais portas e logs verificar?",
    "Como verificar e reiniciar os serviços do Autodesk Wiretap e Stone+Wire no terminal Linux?",
    "O que é uma Timeline no Avid Media Composer e como organizar faixas de vídeo e áudio?",
    "Quais são as boas práticas de configuração de rede para tráfego de áudio Dante sobre IP?",
    "Como diagnosticar lentidão de leitura de mídia em armazenamento compartilhado nas estações de edição Linux?",
    "Crie um script rápido em Bash para testar se a porta TCP 7100 do Wiretap e UDP 4172 do PCoIP estão ativas.",
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
            "options": {"temperature": 0.1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    answer = resp.json().get("message", {}).get("content", "").strip()
    return {"answer": answer, "sources": []}


def check_protocol_score(answer: str) -> int:
    """Conta quantas seções estruturadas de Troubleshooting aparecem na resposta (0 a 5)."""
    upper = answer.upper()
    return sum(1 for sec in STRUCTURED_SECTIONS if sec in upper)


def run_benchmark_mode(
    mode_title: str,
    model_name: str,
    runner_fn: Callable[[str], Dict],
    questions: List[str],
) -> Dict:
    print(f"\n{'=' * 80}")
    print(f"🚀 MODO: {mode_title}")
    print(f"{'=' * 80}")

    warmup_ollama_model(model_name)

    items = []
    total_time = 0.0
    total_chars = 0
    total_protocol_hits = 0

    for idx, q in enumerate(questions, start=1):
        short_q = q[:68] + "..." if len(q) > 68 else q
        print(f"\n  [{idx}/{len(questions)}] {short_q}")
        t0 = time.time()
        try:
            res = runner_fn(q)
            elapsed = time.time() - t0
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            chars = len(answer)
            chars_per_sec = chars / elapsed if elapsed > 0 else 0.0
            proto_score = check_protocol_score(answer)

            total_time += elapsed
            total_chars += chars
            total_protocol_hits += proto_score

            print(
                f"     ⏱️  Tempo: {elapsed:6.2f}s | ⚡ {chars_per_sec:5.1f} chars/s | "
                f"📄 {chars:4d} chars | 📚 Fontes: {len(sources)} | 🎯 Estrutura: {proto_score}/5"
            )

            items.append({
                "question": q,
                "time_sec": round(elapsed, 2),
                "chars": chars,
                "chars_per_sec": round(chars_per_sec, 1),
                "sources_count": len(sources),
                "protocol_score": proto_score,
                "answer_preview": answer[:300],
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"     ❌ Erro após {elapsed:.2f}s: {e}")
            items.append({
                "question": q,
                "time_sec": round(elapsed, 2),
                "error": str(e),
            })

    count = len(questions) or 1
    avg_time = total_time / count
    avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0.0
    avg_proto = total_protocol_hits / count

    print(f"\n  📊 Resumo — {mode_title}:")
    print(f"     • Tempo Médio por Resposta : {avg_time:.2f}s")
    print(f"     • Tempo Total Acumulado    : {total_time:.2f}s")
    print(f"     • Vazão Média de Geração   : {avg_chars_per_sec:.1f} chars/s")
    print(f"     • Aderência ao Protocolo   : {avg_proto:.1f}/5 seções técnicas")

    return {
        "title": mode_title,
        "model": model_name,
        "avg_time_sec": round(avg_time, 2),
        "total_time_sec": round(total_time, 2),
        "avg_chars_per_sec": round(avg_chars_per_sec, 1),
        "avg_protocol_score": round(avg_proto, 2),
        "results": items,
    }


def print_comparison_table(
    ft_direct: Dict,
    ft_rag: Dict,
    base_rag: Dict,
    questions: List[str],
):
    print("\n" + "=" * 95)
    print("🏆 RELATÓRIO COMPARATIVO FINAL: FINE-TUNING (focus-media-solver) vs RAG (gemma3:12b)")
    print("=" * 95)

    header = (
        f"{'Pergunta':<42} | {'FT Direto':>10} | {'FT + RAG':>10} | "
        f"{'Gemma3 RAG':>10} | {'Speedup FT':>11}"
    )
    print("\n" + header)
    print("-" * 95)

    for i, q in enumerate(questions):
        t_ft = ft_direct["results"][i].get("time_sec", 0.0)
        t_hyb = ft_rag["results"][i].get("time_sec", 0.0)
        t_gem = base_rag["results"][i].get("time_sec", 0.0)

        speedup = (t_gem / t_ft) if t_ft > 0 else 0.0
        label = q[:40] + ".." if len(q) > 40 else q
        print(
            f"{label:<42} | {t_ft:>9.2f}s | {t_hyb:>9.2f}s | "
            f"{t_gem:>9.2f}s | {speedup:>9.2f}x 🚀"
        )

    print("-" * 95)
    avg_ft = ft_direct["avg_time_sec"]
    avg_hyb = ft_rag["avg_time_sec"]
    avg_gem = base_rag["avg_time_sec"]
    overall_speedup_direct = (avg_gem / avg_ft) if avg_ft > 0 else 0.0
    overall_speedup_hybrid = (avg_gem / avg_hyb) if avg_hyb > 0 else 0.0

    print(
        f"{'MÉDIA POR PERGUNTA':<42} | {avg_ft:>9.2f}s | {avg_hyb:>9.2f}s | "
        f"{avg_gem:>9.2f}s | {overall_speedup_direct:>9.2f}x ⚡"
    )
    print(
        f"{'TEMPO TOTAL (' + str(len(questions)) + ' PERGUNTAS)':<42} | "
        f"{ft_direct['total_time_sec']:>9.2f}s | {ft_rag['total_time_sec']:>9.2f}s | "
        f"{base_rag['total_time_sec']:>9.2f}s |"
    )
    print(
        f"{'VAZÃO MÉDIA (CHARS/SEGUNDO)':<42} | "
        f"{ft_direct['avg_chars_per_sec']:>8.1f}/s | {ft_rag['avg_chars_per_sec']:>8.1f}/s | "
        f"{base_rag['avg_chars_per_sec']:>8.1f}/s |"
    )
    print("=" * 95)

    print("\n📌 CONCLUSÃO DE PERFORMANCE:")
    if avg_ft > 0 and avg_gem > 0:
        pct_faster_direct = ((avg_gem - avg_ft) / avg_gem) * 100
        print(
            f"  1. Fine-Tuning Puro ({ft_direct['model']}) é {overall_speedup_direct:.2f}x mais rápido "
            f"({pct_faster_direct:.1f}% de redução de latência) que o RAG com {base_rag['model']}."
        )
    if avg_hyb > 0 and avg_gem > 0:
        pct_faster_hybrid = ((avg_gem - avg_hyb) / avg_gem) * 100
        print(
            f"  2. Fine-Tuning + RAG Híbrido ({ft_rag['model']} + Qdrant) é {overall_speedup_hybrid:.2f}x mais rápido "
            f"({pct_faster_hybrid:.1f}% de redução de latência) que o RAG com {base_rag['model']}."
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
    args = parser.parse_args()

    questions = BENCHMARK_QUESTIONS[:args.limit] if args.limit > 0 else BENCHMARK_QUESTIONS

    print("=" * 95)
    print("⚡ BENCHMARK FOCUS AI: FINE-TUNING (focus-media-solver) vs RAG (gemma3:12b)")
    print(f"   Data/Hora : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Perguntas : {len(questions)}")
    print("=" * 95)

    # 1. Fine-Tuning Direto (Sem busca no Qdrant)
    ft_direct_stats = run_benchmark_mode(
        mode_title=f"1. FINE-TUNING PURO ({args.ft_model} direto, sem busca RAG)",
        model_name=args.ft_model,
        runner_fn=lambda q: ask_finetuned_direct(q, args.ft_model),
        questions=questions,
    )

    # 2. Híbrido: Fine-Tuning + RAG V2 (Qdrant + focus-media-solver)
    ft_rag_stats = run_benchmark_mode(
        mode_title=f"2. HÍBRIDO: FINE-TUNING + RAG V2 ({args.ft_model} + Qdrant)",
        model_name=args.ft_model,
        runner_fn=lambda q: ask_rag_v2(q, model=args.ft_model),
        questions=questions,
    )

    # 3. RAG Tradicional (Qdrant + gemma3:12b)
    base_rag_stats = run_benchmark_mode(
        mode_title=f"3. RAG TRADICIONAL ({args.rag_model} + Qdrant)",
        model_name=args.rag_model,
        runner_fn=lambda q: ask_rag_v2(q, model=args.rag_model),
        questions=questions,
    )

    print_comparison_table(ft_direct_stats, ft_rag_stats, base_rag_stats, questions)

    # Salva relatório JSON
    out_dir = PROJECT_V2_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_ft_vs_rag_{ts}.json"

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions_count": len(questions),
        "ft_direct": ft_direct_stats,
        "ft_rag_hybrid": ft_rag_stats,
        "rag_baseline": base_rag_stats,
    }
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Relatório completo salvo em: {out_file}\n")


if __name__ == "__main__":
    main()
