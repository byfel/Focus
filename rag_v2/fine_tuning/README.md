# FOCUS AI — Guia de Fine-Tuning Incremental & Pipeline de 5 Etapas

Este módulo especializa Large Language Models (LLMs) para atuarem como **Engenheiros Sêniores de Mídia, Broadcast e Pós-Produção**, aprendendo continuamente com **todos os documentos da base** e atualizando o modelo sempre que **novos documentos** entram no processo de ingestão.

O pipeline foi projetado especificamente para o servidor com **8x NVIDIA Quadro M6000 12GB (Maxwell sm_52)**, utilizando **FP16** e atenção nativa para máxima estabilidade e performance.

---

## 🔄 Fluxo Completo das 5 Etapas (Para Novos Documentos)

Sempre que você adicionar novos PDFs ou manuais em `/opt/ai/rag/documents/`, o processo completo agora possui **5 etapas** (todas incrementais, processando apenas o que mudou ou o que é novo):

```bash
# Etapa 1: Extração de texto e Vision dos PDFs novos/alterados -> chunks.json
python3 /opt/ai/rag/scripts/ingest.py --scope all

# Etapa 2: Geração incremental de embeddings V1 -> embeddings.json
python3 /opt/ai/rag/scripts/embed.py

# Etapa 3: Sincronização incremental com Qdrant V1 -> rag_documents
python3 /opt/ai/rag/scripts/migrate_to_qdrant.py

# Etapa 4: População incremental do Qdrant V2 (nomic-embed-text) -> rag_documents_v2
python3 /opt/ai/rag_v2/scripts/populate_v2.py

# Etapa 5 (NOVO!): Aprendizado e Fine-Tuning Incremental dos novos documentos + Deploy no Ollama
python3 /opt/ai/rag_v2/scripts/train_new_docs.py --chunks-per-doc 8 --num-gpus 8
```

> **💡 Dica:** Você também pode disparar a Etapa 5 direto pelo `populate_v2.py` usando `--train`:
> ```bash
> python3 /opt/ai/rag_v2/scripts/populate_v2.py --train
> ```
> Ou rodar **todas as 5 etapas com um único comando**:
> ```bash
> python3 /opt/ai/rag_v2/scripts/run_full_pipeline.py --scope all --num-gpus 8
> ```

---

## 🧠 Como Funciona o Aprendizado de Todos os Documentos

Diferente de uma leitura sequencial simples que ficaria presa apenas no primeiro PDF de `chunks.json`, o [`generate_dataset.py`](file:///c:/Users/Felipe/Documents/Focus-main/rag_v2/fine_tuning/generate_dataset.py) agora:

1. **Agrupa por Documento e Prioriza Densidade Técnica:** Seleciona trechos distribuídos do início, meio e fim de **cada documento** (`--chunks-per-doc 8` por padrão, ou `0` para processar 100% dos chunks de cada documento), ignorando sumários e capas vazias.
2. **Ordem Intercalada (Round-Robin):** Processa 1 chunk do Documento A, 1 do Documento B, 1 do Documento C... garantindo que **todos os manuais da sua base** estejam representados no treinamento.
3. **Estado Incremental (`dataset/dataset_state.json`):** Rastreia o `file_hash` (SHA256) e os `chunk_id`s já aprendidos de cada PDF:
   - Documentos que já atingiram a meta aparecem como `✅ JÁ APRENDIDO` e são pulados instantaneamente.
   - Documentos recém-adicionados no `ingest.py` aparecem como `🆕 NOVO DOCUMENTO` e têm seus casos gerados e adicionados ao dataset.
4. **Salvamento a cada Chunk (Checkpoint):** Cada exemplo gerado é gravado imediatamente em `dataset/media_solver_train.jsonl`. Se você interromper com `Ctrl+C`, nada é perdido!

---

## 🛠️ Comandos Úteis do Gerador e Treinador

### Aproveitando o primeiro dataset que já rodou no servidor
Como sua primeira execução já processou o documento `"Criação de Produto no Diva V2.pdf"`, ao atualizar o script você pode marcá-lo como já aprendido para que o gerador vá direto aprender todos os **outros** documentos da base (mantendo todos os exemplos que já foram gerados no `.jsonl`):

```bash
cd /opt/ai/rag_v2/fine_tuning
python3 generate_dataset.py --source all \
    --mark-learned "Criação de Produto no Diva V2.pdf" \
    --chunks-per-doc 8
```

### Processar apenas documentos 100% novos
```bash
python3 generate_dataset.py --source all --only-new-docs --chunks-per-doc 8
```

### Treinamento Incremental LoRA (`train.py`)
O [`train.py`](file:///c:/Users/Felipe/Documents/Focus-main/rag_v2/fine_tuning/train.py) verifica automaticamente se o `media_solver_train.jsonl` recebeu novos exemplos desde o último treino (`output/train_state.json`) e **continua refinando o adaptador LoRA existente** (`output/lora_adapter`) sem perder o conhecimento anterior:

```bash
# Em 1 GPU:
python3 train.py --epochs 3

# Nas 8 GPUs Quadro M6000:
accelerate launch --multi_gpu --num_processes 8 train.py --epochs 3 --batch-size 1 --grad-accum 4
```
