import os
import json
import time
import base64
import requests
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL = "gemma3:4b"

IMAGE_DIR = Path("/tmp")

OUTPUT_DIR = Path("/opt/ai/rag/vision_results")

OLLAMA_URL = "http://localhost:11434/api/chat"

NUM_PREDICT = 600


PROMPT = """Analise esta imagem de um manual técnico para indexação em RAG.

Extraia as informações visuais relevantes para responder perguntas futuras sobre esta tela.

Inclua:
- software ou sistema;
- menus e botões;
- campos e opções;
- nomes, textos e valores visíveis;
- datas e números;
- procedimento demonstrado;
- sequência de ações indicada pelas marcações.

Preserve nomes e textos importantes exatamente como aparecem.

Não invente informações.
Não faça introdução.
Não repita informações.

Retorne em tópicos objetivos."""


# ============================================================
# PREPARAÇÃO
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


images = sorted(
    IMAGE_DIR.glob("logagem_img_*.png"),
    key=lambda p: p.name
)

if not images:
    print("ERRO: nenhuma imagem encontrada em:")
    print(IMAGE_DIR)
    print("Esperado: /tmp/logagem_img_*.png")
    exit(1)


print("=" * 70)
print("TESTE DE VISION - QWEN3-VL")
print("=" * 70)

print(f"Modelo:       {MODEL}")
print(f"Imagens:      {len(images)}")
print(f"Output:       {OUTPUT_DIR}")
print("=" * 70)


# ============================================================
# PROCESSAMENTO
# ============================================================

results = []

total_start = time.time()

for index, image_path in enumerate(images, start=1):

    print()
    print("=" * 70)
    print(f"[{index}/{len(images)}] Processando: {image_path.name}")
    print("=" * 70)

    start = time.time()

    try:

        # ----------------------------------------------------
        # BASE64
        # ----------------------------------------------------

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")


        # ----------------------------------------------------
        # REQUEST
        # ----------------------------------------------------

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_b64]
                }
            ],
            "stream": False,
            "options": {
                "num_predict": NUM_PREDICT
            }
        }


        # ----------------------------------------------------
        # OLLAMA
        # ----------------------------------------------------

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=3600
        )

        response.raise_for_status()

        data = response.json()


        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        content = data.get("message", {}).get("content", "")

        elapsed = time.time() - start


        result = {
            "image": image_path.name,
            "model": MODEL,
            "elapsed_seconds": round(elapsed, 2),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "content": content
        }


        results.append(result)


        # ----------------------------------------------------
        # SALVAR INDIVIDUAL
        # ----------------------------------------------------

        output_file = OUTPUT_DIR / f"{image_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2
            )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()
        print(content)

        print()
        print("--- ESTATÍSTICAS ---")
        print(f"Tempo:             {elapsed:.2f}s")
        print(f"Tokens gerados:    {data.get('eval_count')}")
        print(f"Tokens prompt:     {data.get('prompt_eval_count')}")
        print(f"Arquivo salvo:     {output_file}")


    except Exception as e:

        elapsed = time.time() - start

        error_result = {
            "image": image_path.name,
            "model": MODEL,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e)
        }

        results.append(error_result)

        output_file = OUTPUT_DIR / f"{image_path.stem}_ERROR.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                error_result,
                f,
                ensure_ascii=False,
                indent=2
            )

        print()
        print("ERRO:")
        print(e)


# ============================================================
# RESUMO
# ============================================================

total_elapsed = time.time() - total_start

summary = {
    "model": MODEL,
    "num_images": len(images),
    "num_predict": NUM_PREDICT,
    "total_elapsed_seconds": round(total_elapsed, 2),
    "results": results
}


summary_file = OUTPUT_DIR / "summary.json"

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print()
print("=" * 70)
print("RESUMO FINAL")
print("=" * 70)

for result in results:

    print(
        f"{result['image']:<30} "
        f"{result.get('elapsed_seconds', 0):>8.2f}s "
        f"{result.get('eval_count', '-'):>6} tokens"
    )

print("-" * 70)

print(f"Imagens processadas: {len(images)}")
print(f"Tempo total:         {total_elapsed:.2f}s")
print(f"Resumo:              {summary_file}")

print("=" * 70)
