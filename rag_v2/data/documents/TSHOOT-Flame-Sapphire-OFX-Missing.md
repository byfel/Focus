# TSHOOT: Sapphire Não Aparece no Autodesk Flame (Link Quebrado & Migração Sparks para OFX)

- **ID do Documento:** `KB-FLAME-SAPPHIRE-001`
- **Host:** Autodesk Flame 2025.2.x (Linux x86_64, Rocky/CentOS)
- **Plugin:** Boris FX Sapphire OFX 2025 (`SapphireOFX-2025.000-1.0.x86_64`)
- **Severidade:** Alta (Plugin ausente na interface do Flame)
- **Status:** Resolvido / Validado

---

## 1. Resumo Executivo
O Boris FX Sapphire não aparecia no Autodesk Flame 2025 por dois motivos principais:
1. **Transição Arquitetural (Sparks -> OFX):** A Boris FX descontinuou a arquitetura legada "Sparks" a partir da versão 2020. No Flame 2025, o plugin é estritamente **OFX (Open FX)** e não deve ser procurado na pasta de Sparks da Autodesk (`presets/.../sparks/`).
2. **Broken Symlink (Link Quebrado):** O link simbólico global em `/usr/OFX/Plugins/Sapphire.ofx.bundle` apontava para `/opt/BorisFX/SapphireOFX/plugins/Sapphire.ofx.bundle` (com uma subpasta `/plugins/` inexistente). O comando `ls` exibiu o arquivo em **vermelho** e o Flame não carregava o bundle.

---

## 2. Sintomas e Comportamento
- Os nós do Sapphire (ex: `S_Glow`, `S_Blur`, `S_LensFlare`) não aparecem no Batch, Node Bin ou busca rápida do Flame.
- Ao inspecionar os diretórios de Sparks do Flame (`presets/2025.2.7/sparks/`), apenas os sparks básicos da Autodesk estão presentes.
- No terminal Linux, ao executar `ls -la /usr/OFX/Plugins/`, o link `Sapphire.ofx.bundle` aparece destacado em **vermelho** (indicador de link órfão / broken symlink).

---

## 3. Causa Raiz (Root Cause Analysis)
1. **Diferença de Formatos:** O pacote legado `genarts-sapphire-sparks11` ainda estava presente no sistema, causando confusão quanto à localização do plugin. O Flame moderno utiliza o pacote `SapphireOFX-2025`.
2. **Caminho Alvo Incorreto:** O link simbólico apontava para:
   `/opt/BorisFX/SapphireOFX/plugins/Sapphire.ofx.bundle`  *(inválido)*
   Em vez do caminho físico real instalado pelo RPM:
   `/opt/BorisFX/SapphireOFX/Sapphire.ofx.bundle`  *(válido)*
3. **Cache de Plugins do Flame:** O Flame armazena em cache o estado dos plugins OFX no arquivo `ofxPluginCache.xml`. Mesmo após corrigir o link, o Flame mantém o estado ignorado até que o cache seja apagado.

---

## 4. Diagnóstico Técnico

### 4.1. Verificação de Pacotes RPM
```bash
rpm -qa | grep -i sapphire
```
*Saída esperada:*
- `genarts-sapphire-sparks11-11.020-1.x86_64` (Legado Sparks, não compatível com Flame 2025)
- `SapphireOFX-2025.000-1.0.x86_64` (OFX moderno correto)

### 4.2. Inspeção do Diretório OFX
```bash
ls -la /usr/OFX/Plugins/
```
*Saída observada:*
`Sapphire.ofx.bundle -> /opt/BorisFX/SapphireOFX/plugins/Sapphire.ofx.bundle` (em vermelho)

### 4.3. Identificação do Caminho Real no RPM
```bash
rpm -ql SapphireOFX-2025.000-1.0.x86_64 | grep -i "\.bundle" | head -n 3
```
*Resultado real:*
`/opt/BorisFX/SapphireOFX/Sapphire.ofx.bundle`

---

## 5. Procedimento de Resolução (Action Plan)

### Passo 1: Recriar o link simbólico correto
```bash
ln -sfn /opt/BorisFX/SapphireOFX/Sapphire.ofx.bundle /usr/OFX/Plugins/Sapphire.ofx.bundle
```

### Passo 2: Validar o link no terminal
```bash
ls -la /usr/OFX/Plugins/
```
*O item deve aparecer em azul/ciano, confirmando que o destino é válido.*

### Passo 3: Limpar o cache OFX do Autodesk Flame
```bash
rm -f /opt/Autodesk/flame_2025.2.7/status/ofxPluginCache.xml
rm -f /opt/Autodesk/flame_2025.2.6/status/ofxPluginCache.xml
```

### Passo 4: Validação no Flame
1. Abra o Flame e acesse o módulo **Batch**.
2. Abra a busca rápida de nós (espaço) e pesquise por `S_Glow`, `S_Blur`, `S_LensFlare`.
3. Alternativamente, arraste o nó **OFX** e selecione Boris FX na lista de efeitos.

---

## 6. Metadados para RAG / Focus AI
- **Palavras-chave:** autodesk-flame, flame-2025, sapphire, boris-fx, ofx, sparks, broken-symlink, dircolors-vermelho, linux, ofxPluginCache.xml.
- **Perguntas Respondidas:**
  - "Por que o Sapphire não aparece no Flame 2025?"
  - "O que significa o arquivo Sapphire.ofx.bundle em vermelho no terminal?"
  - "Onde fica o diretório de instalação do Sapphire OFX no Linux?"
  - "Como forçar o Flame a atualizar ou recarregar os plugins OFX?"
