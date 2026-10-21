#!/usr/bin/env python3
"""
FOCUS AI - Exportador & Deploy para Ollama.

Realiza a fusão (merge) dos pesos do adaptador LoRA com o modelo base Hugging Face,
cria o Modelfile com a persona do Solucionador de Mídia e registra o novo modelo no Ollama.

Uso:
    python3 export_to_ollama.py --model-name focus-media-solver
"""

import os
import sys
import shutil
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_ollama")

MODELFILE_TEMPLATE = """# FOCUS AI - Solucionador de Mídia & Broadcast
FROM {model_source}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>\"\"\"

SYSTEM \"\"\"Você é o FOCUS AI - Solucionador Sênior e Especialista em Engenharia de Mídia, Broadcast, Áudio/Vídeo Profissional, Infraestrutura e Automação de Estúdios.

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
Responda sempre em português técnico, objetivo e com alto rigor de engenharia.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""


def merge_lora_weights(base_model_name: str, adapter_dir: Path, output_merged_dir: Path):
    """Funde os adaptadores LoRA com o modelo base e salva em formato HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Carregando modelo base: {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU para merge seguro sem estourar VRAM
        trust_remote_code=True,
    )

    logger.info(f"Carregando adaptador LoRA de: {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    logger.info("Fundindo pesos LoRA (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    logger.info(f"Salvando modelo fundido em: {output_merged_dir}...")
    output_merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_merged_dir))

    logger.info("Salvando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_merged_dir))

    logger.info("✅ Fusão concluída com sucesso!")


def create_modelfile(model_source_path: str, modelfile_path: Path):
    """Gera o arquivo Modelfile para o Ollama."""
    content = MODELFILE_TEMPLATE.format(model_source=model_source_path)
    modelfile_path.write_text(content, encoding="utf-8")
    logger.info(f"📄 Modelfile gerado em: {modelfile_path}")


def register_in_ollama(model_name: str, modelfile_path: Path):
    """Executa 'ollama create <model_name> -f <modelfile>'."""
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    logger.info(f"Executando: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Saída do Ollama:\n{res.stdout}")
        logger.info(f"🎉 Modelo '{model_name}' registrado com sucesso no Ollama!")
    except FileNotFoundError:
        logger.warning("O comando 'ollama' não foi encontrado no PATH. Execute manualmente:")
        logger.warning(f"  ollama create {model_name} -f {modelfile_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao registrar no Ollama: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="FOCUS AI - Merge e Deploy no Ollama")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Nome do modelo base no HuggingFace")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Diretório onde estão os pesos LoRA (padrão: output/lora_adapter)")
    parser.add_argument("--merged-dir", type=str, default=None,
                        help="Diretório onde salvar o modelo fundido (padrão: output/merged_model)")
    parser.add_argument("--model-name", type=str, default="focus-media-solver",
                        help="Nome com o qual o modelo será registrado no Ollama")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Pular etapa de fusão (usar se já tiver o modelo fundido ou GGUF)")
    parser.add_argument("--gguf-path", type=str, default=None,
                        help="Caminho direto para arquivo .gguf se já convertido")
    args = parser.parse_args()

    root_ft = Path(__file__).resolve().parent
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else root_ft / "output" / "lora_adapter"
    merged_dir = Path(args.merged_dir) if args.merged_dir else root_ft / "output" / "merged_model"
    modelfile_path = root_ft / "Modelfile"

    if not args.skip_merge and not args.gguf_path:
        if not adapter_dir.exists():
            logger.error(f"Adaptador LoRA não encontrado em: {adapter_dir}")
            logger.error("Execute primeiro o treinamento com 'python3 train.py'")
            sys.exit(1)
        merge_lora_weights(args.base_model, adapter_dir, merged_dir)

    # Fonte para o Modelfile
    model_source = args.gguf_path if args.gguf_path else str(merged_dir)

    create_modelfile(model_source, modelfile_path)
    register_in_ollama(args.model_name, modelfile_path)


if __name__ == "__main__":
    main()
