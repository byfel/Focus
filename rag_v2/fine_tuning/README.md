# FOCUS AI — Guia de Fine-Tuning: Solucionador de Mídia & Broadcast

Este módulo especializa Large Language Models (LLMs) para atuarem como **Engenheiros Sêniores de Mídia, Broadcast e Pós-Produção**, ensinando a metodologia rigorosa de Troubleshooting Técnico (Root Cause Analysis - RCA, identificação de Broken Symlinks, análise de pacotes RPM, validação de rotas Dante, inspeção de logs de PCoIP e automações em Bash/Python).

O pipeline foi projetado especificamente para o servidor com **8x NVIDIA Quadro M6000 12GB (Maxwell sm_52)**, utilizando **FP16** e atenção nativa para máxima estabilidade e performance.

---

## 📁 Estrutura de Arquivos

```
rag_v2/fine_tuning/
├── dataset/
│   └── media_solver_train.jsonl    # Dataset supervisionado gerado (ChatML)
├── generate_dataset.py             # Sintetizador: extrai manuais e cria casos de treino
├── train.py                        # Script de treino LoRA (SFTTrainer + Accelerate Multi-GPU)
├── export_to_ollama.py             # Merge LoRA -> Cria Modelfile -> Registra no Ollama
├── requirements.txt                # Dependências Python para o servidor
└── README.md                       # Este guia
```

---

## 🚀 Passo a Passo no Servidor

### 1. Instalar as Dependências no Servidor

No ambiente virtual do seu servidor Linux com as GPUs:

```bash
cd /opt/ai/rag_v2/fine_tuning
pip install -r requirements.txt
```

---

### 2. Gerar o Dataset de Treinamento

O gerador pode extrair casos de três fontes:
1. **Documentos e KBs locais** (`/opt/ai/rag_v2/data/documents/` ou `/opt/ai/rag/documents/`)
2. **Chunks já indexados no Qdrant** (`rag_documents_v2`)
3. **Casos Ouro de Broadcast pré-configurados** (Flame, Sapphire OFX, Dante, PCoIP, etc.)

#### Opção A: Gerar a partir dos Manuais e KBs locais (Recomendado):
```bash
python3 generate_dataset.py --source files --include-seeds --max-samples 100
```

#### Opção B: Gerar a partir do Qdrant:
```bash
python3 generate_dataset.py --source qdrant --max-samples 100
```

#### Opção C: Gerar apenas os Casos Semente imediatamente (sem chamar o LLM):
```bash
python3 generate_dataset.py --source seeds-only
```

O arquivo final será salvo em `dataset/media_solver_train.jsonl`.

---

### 3. Executar o Treinamento Multi-GPU (8x Quadro M6000)

O script `train.py` usa **LoRA** em **FP16** (ideal para arquitetura Maxwell):

#### Execução rápida em 1 GPU (para teste de sanidade):
```bash
python3 train.py --epochs 3 --batch-size 1 --grad-accum 8
```

#### Execução completa distribuída nas 8 GPUs (via Accelerate / DDP):
```bash
accelerate launch --multi_gpu --num_processes 8 train.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --epochs 5 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 2e-4
```

> **Nota Técnica para Quadro M6000:**  
> - `fp16=True` e `bf16=False` já estão fixados no script.  
> - `attn_implementation="eager"` garante compatibilidade com Compute Capability 5.2.

Os pesos do adaptador serão salvos em `output/lora_adapter/`.

---

### 4. Fundir os Pesos e Publicar no Ollama

Após o término do treinamento, execute o exportador:

```bash
python3 export_to_ollama.py --model-name focus-media-solver
```

O script automaticamente:
1. Funde os pesos LoRA com o modelo base (`merge_and_unload`).
2. Gera o `Modelfile` contendo o System Prompt especialista em Broadcast.
3. Executa `ollama create focus-media-solver -f Modelfile`.

Para testar no terminal:
```bash
ollama run focus-media-solver "O plugin Sapphire não aparece no Flame 2025 no Rocky Linux, o que fazer?"
```

---

### 5. Utilizar no FOCUS AI (RAG v2)

O modelo `focus-media-solver` já foi adicionado aos `SUPPORTED_MODELS` no arquivo [`core/config.py`](file:///c:/Users/Felipe/Documents/Focus-main/rag_v2/core/config.py).

Basta reiniciar a API Web do FOCUS ou selecioná-lo no menu dropdown de modelos na interface web!
