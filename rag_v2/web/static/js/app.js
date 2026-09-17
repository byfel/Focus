// FOCUS AI v2 - Client Application JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentConversationId = null;
    let conversations = [];
    let supportedModels = [];
    let selectedModel = 'gemma3:12b';
    let isGenerating = false;
    let timerInterval = null;

    // DOM Elements
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const modelSelect = document.getElementById('modelSelect');
    const conversationList = document.getElementById('conversationList');
    const messagesContainer = document.getElementById('messagesContainer');
    const messagesContent = document.getElementById('messagesContent');
    const chatHeaderTitle = document.getElementById('chatHeaderTitle');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const currentUserDisplay = document.getElementById('currentUserDisplay');

    // Initialize
    init();

    async function init() {
        setupEventListeners();
        await loadCurrentUser();
        await loadModels();
        await loadConversations();
    }

    function setupEventListeners() {
        // Sidebar toggle
        if (toggleSidebarBtn) {
            toggleSidebarBtn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
            });
        }

        // New Chat
        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => {
                startNewChat();
            });
        }

        // Model selector change
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                selectedModel = e.target.value;
            });
        }

        // Auto-expanding textarea
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
                updateSendButtonState();
            });

            // Enter to send, Shift+Enter for new line
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!isGenerating && chatInput.value.trim().length > 0) {
                        sendMessage();
                    }
                }
            });
        }

        // Send button click
        if (sendBtn) {
            sendBtn.addEventListener('click', () => {
                if (!isGenerating && chatInput.value.trim().length > 0) {
                    sendMessage();
                }
            });
        }

        // Logout
        if (logoutBtn) {
            logoutBtn.addEventListener('click', async () => {
                try {
                    await fetch('/logout', { method: 'POST' });
                    window.location.href = '/';
                } catch (e) {
                    window.location.href = '/';
                }
            });
        }
    }

    function updateSendButtonState() {
        const hasText = chatInput.value.trim().length > 0;
        sendBtn.disabled = !hasText || isGenerating;
    }

    async function loadCurrentUser() {
        try {
            const res = await fetch('/me');
            if (res.ok) {
                const data = await res.json();
                if (currentUserDisplay) {
                    currentUserDisplay.textContent = data.username || 'admin';
                }
            } else {
                window.location.href = '/';
            }
        } catch (e) {
            console.error('Erro ao obter usuário atual:', e);
        }
    }

    async function loadModels() {
        try {
            const res = await fetch('/models');
            if (res.ok) {
                const data = await res.json();
                supportedModels = data.models || [];
                modelSelect.innerHTML = '';
                
                supportedModels.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === 'gemma3:12b') opt.selected = true;
                    modelSelect.appendChild(opt);
                });

                if (modelSelect.value) {
                    selectedModel = modelSelect.value;
                }
            }
        } catch (e) {
            console.error('Erro ao carregar modelos:', e);
        }
    }

    async function loadConversations() {
        try {
            const res = await fetch('/conversations');
            if (res.ok) {
                const data = await res.json();
                conversations = data.conversations || [];
                renderConversations();

                if (conversations.length > 0 && !currentConversationId) {
                    // Open the most recent conversation or show empty welcome
                    selectConversation(conversations[0].id);
                } else if (conversations.length === 0) {
                    renderWelcomeHero();
                }
            }
        } catch (e) {
            console.error('Erro ao carregar conversas:', e);
        }
    }

    function renderConversations() {
        conversationList.innerHTML = '';

        if (conversations.length === 0) {
            const emptyNotice = document.createElement('div');
            emptyNotice.style.padding = '12px 14px';
            emptyNotice.style.fontSize = '12px';
            emptyNotice.style.color = 'var(--text-muted)';
            emptyNotice.textContent = 'Nenhuma conversa ainda.';
            conversationList.appendChild(emptyNotice);
            return;
        }

        const sectionTitle = document.createElement('div');
        sectionTitle.className = 'history-section-title';
        sectionTitle.textContent = 'Histórico Recente';
        conversationList.appendChild(sectionTitle);

        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = `conversation-item ${conv.id === currentConversationId ? 'active' : ''}`;
            item.dataset.id = conv.id;

            const titleWrapper = document.createElement('div');
            titleWrapper.className = 'conversation-title-wrapper';

            titleWrapper.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span class="conv-title" title="${escapeHtml(conv.title)}">${escapeHtml(conv.title)}</span>
            `;

            const actions = document.createElement('div');
            actions.className = 'conversation-actions';

            // Rename button
            const renameBtn = document.createElement('button');
            renameBtn.className = 'action-icon-btn';
            renameBtn.title = 'Renomear';
            renameBtn.innerHTML = `
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 20h9"/>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                </svg>
            `;
            renameBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                renameConversation(conv.id, conv.title);
            });

            // Delete button
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'action-icon-btn delete';
            deleteBtn.title = 'Excluir';
            deleteBtn.innerHTML = `
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
            `;
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(conv.id);
            });

            actions.appendChild(renameBtn);
            actions.appendChild(deleteBtn);

            item.appendChild(titleWrapper);
            item.appendChild(actions);

            item.addEventListener('click', () => {
                selectConversation(conv.id);
            });

            conversationList.appendChild(item);
        });
    }

    async function selectConversation(id) {
        currentConversationId = id;
        renderConversations();

        try {
            const res = await fetch(`/conversation/${id}`);
            if (res.ok) {
                const data = await res.json();
                chatHeaderTitle.textContent = data.title || 'Conversa';
                renderMessages(data.messages || []);
            }
        } catch (e) {
            console.error('Erro ao carregar detalhes da conversa:', e);
        }
    }

    async function startNewChat() {
        currentConversationId = null;
        chatHeaderTitle.textContent = 'Nova conversa';
        renderWelcomeHero();
        renderConversations();
        chatInput.focus();
    }

    function renderWelcomeHero() {
        messagesContent.innerHTML = `
            <div class="welcome-hero">
                <div class="welcome-logo-orb">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                        <circle cx="12" cy="12" r="3.5" fill="#FFFFFF" stroke="none"/>
                    </svg>
                </div>
                <h1 class="welcome-title">Como posso apoiar a Engenharia hoje?</h1>
                <p class="welcome-subtitle">
                    Assistente RAG v2 indexado com <strong>nomic-embed-text</strong> para consulta técnica em tempo real aos manuais de Flame, Media Composer, Dante, PCoIP e Linux.
                </p>

                <div class="prompt-suggestions-grid">
                    <div class="suggestion-card" data-prompt="O que é rede Dante e como funciona o roteamento de áudio?">
                        <div class="suggestion-card-header">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                            </svg>
                            <span>Áudio & Redes</span>
                        </div>
                        <div class="suggestion-card-text">O que é rede Dante e como funciona o roteamento de áudio?</div>
                    </div>

                    <div class="suggestion-card" data-prompt="Como configurar e solucionar problemas de latência no protocolo PCoIP?">
                        <div class="suggestion-card-header">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                                <line x1="8" y1="21" x2="16" y2="21"/>
                                <line x1="12" y1="17" x2="12" y2="21"/>
                            </svg>
                            <span>PCoIP / Remote</span>
                        </div>
                        <div class="suggestion-card-text">Como configurar e solucionar problemas de latência no protocolo PCoIP?</div>
                    </div>

                    <div class="suggestion-card" data-prompt="Quais são os requisitos de hardware e passos para instalar o Flame no Linux?">
                        <div class="suggestion-card-header">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                                <polyline points="2 17 12 22 22 17"/>
                                <polyline points="2 12 12 17 22 12"/>
                            </svg>
                            <span>Autodesk Flame</span>
                        </div>
                        <div class="suggestion-card-text">Quais são os requisitos de hardware e passos para instalar o Flame no Linux?</div>
                    </div>

                    <div class="suggestion-card" data-prompt="Como funciona o gerenciamento de mídias e Timeline no Avid Media Composer?">
                        <div class="suggestion-card-header">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
                                <line x1="7" y1="2" x2="7" y2="22"/>
                                <line x1="17" y1="2" x2="17" y2="22"/>
                            </svg>
                            <span>Media Composer</span>
                        </div>
                        <div class="suggestion-card-text">Como funciona o gerenciamento de mídias e Timeline no Avid Media Composer?</div>
                    </div>
                </div>
            </div>
        `;

        // Bind suggestion clicks
        document.querySelectorAll('.suggestion-card').forEach(card => {
            card.addEventListener('click', () => {
                const promptText = card.dataset.prompt;
                if (promptText) {
                    chatInput.value = promptText;
                    chatInput.style.height = 'auto';
                    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
                    updateSendButtonState();
                    sendMessage();
                }
            });
        });
    }

    function renderMessages(messages) {
        if (messages.length === 0) {
            renderWelcomeHero();
            return;
        }

        messagesContent.innerHTML = '';
        messages.forEach(msg => {
            appendMessageToUI(msg.role, msg.content, msg.sources, false);
        });

        scrollToBottom();
    }

    function appendMessageToUI(role, content, sources = [], animate = true) {
        // If welcome hero is shown, clear it first
        if (messagesContent.querySelector('.welcome-hero')) {
            messagesContent.innerHTML = '';
        }

        const row = document.createElement('div');
        row.className = `message-row ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        if (role === 'assistant') {
            avatar.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="9"/>
                    <circle cx="12" cy="12" r="3" fill="currentColor"/>
                </svg>
            `;
        } else {
            avatar.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            `;
        }

        const body = document.createElement('div');
        body.className = 'message-body';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = formatMarkdown(content);

        body.appendChild(bubble);

        // Sources section for assistant
        if (role === 'assistant' && sources && sources.length > 0) {
            const sourcesSec = createSourcesElement(sources);
            body.appendChild(sourcesSec);
        }

        if (role === 'user') {
            row.appendChild(body);
            row.appendChild(avatar);
        } else {
            row.appendChild(avatar);
            row.appendChild(body);
        }

        messagesContent.appendChild(row);

        // Bind copy code buttons
        bubble.querySelectorAll('.code-copy-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const code = btn.nextElementSibling.innerText;
                navigator.clipboard.writeText(code).then(() => {
                    btn.textContent = 'Copiado!';
                    setTimeout(() => btn.textContent = 'Copiar', 2000);
                });
            });
        });

        if (animate) {
            scrollToBottom();
        }
    }

    function createSourcesElement(sources) {
        const sec = document.createElement('div');
        sec.className = 'sources-section';

        const header = document.createElement('div');
        header.className = 'sources-header';
        header.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                </svg>
                <span>${sources.length} documento${sources.length > 1 ? 's' : ''} técnico${sources.length > 1 ? 's' : ''} indexado${sources.length > 1 ? 's' : ''}</span>
            </div>
            <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="transition: transform 0.2s;">
                <polyline points="6 9 12 15 18 9"/>
            </svg>
        `;

        const list = document.createElement('div');
        list.className = 'sources-list';
        list.style.display = 'none';

        sources.forEach((s, idx) => {
            const item = document.createElement('div');
            item.className = 'source-item';

            const scorePercent = (s.score ? (s.score * 100).toFixed(1) : 'N/A') + '%';
            const pageText = s.page ? `Pág. ${s.page}` : 'Geral';

            item.innerHTML = `
                <div class="source-item-meta">
                    <span><strong>[${idx + 1}]</strong> ${escapeHtml(s.document || 'Documento')} (${pageText})</span>
                    <span class="source-score-badge">Score: ${scorePercent}</span>
                </div>
                ${s.text ? `<div class="source-snippet">"${escapeHtml(s.text.slice(0, 240))}..."</div>` : ''}
            `;
            list.appendChild(item);
        });

        header.addEventListener('click', () => {
            const isHidden = list.style.display === 'none';
            list.style.display = isHidden ? 'flex' : 'none';
            const chevron = header.querySelector('.chevron-icon');
            if (chevron) {
                chevron.style.transform = isHidden ? 'rotate(180deg)' : 'rotate(0deg)';
            }
        });

        sec.appendChild(header);
        sec.appendChild(list);
        return sec;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;

        isGenerating = true;
        updateSendButtonState();

        // Ensure we have a conversation ID
        if (!currentConversationId) {
            try {
                const res = await fetch('/conversation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: text.slice(0, 40) })
                });
                if (res.ok) {
                    const convData = await res.json();
                    currentConversationId = convData.id;
                    chatHeaderTitle.textContent = convData.title;
                    await loadConversations();
                }
            } catch (e) {
                console.error('Erro ao criar conversa:', e);
            }
        }

        // Add user message to UI
        appendMessageToUI('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add Thinking Indicator with live timer
        const thinkingRow = document.createElement('div');
        thinkingRow.className = 'message-row assistant';
        thinkingRow.id = 'thinkingRow';

        thinkingRow.innerHTML = `
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <circle cx="12" cy="12" r="9"/>
                    <circle cx="12" cy="12" r="3" fill="currentColor"/>
                </svg>
            </div>
            <div class="message-body">
                <div class="thinking-indicator">
                    <div class="shimmer-bar"></div>
                    <span id="thinkingTimerText">Consultando nomic-embed-text e gerando resposta... 0.0s</span>
                </div>
            </div>
        `;
        messagesContent.appendChild(thinkingRow);
        scrollToBottom();

        const startTime = Date.now();
        const timerText = thinkingRow.querySelector('#thinkingTimerText');
        timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            if (timerText) {
                timerText.textContent = `Consultando nomic-embed-text e gerando com ${selectedModel}... ${elapsed}s`;
            }
        }, 100);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    model: selectedModel,
                    conversation_id: currentConversationId
                })
            });

            clearInterval(timerInterval);
            const thinkingElem = document.getElementById('thinkingRow');
            if (thinkingElem) thinkingElem.remove();

            if (response.ok) {
                const data = await response.json();
                appendMessageToUI('assistant', data.answer, data.sources);
                await loadConversations(); // refresh title if updated
            } else {
                const errData = await response.json().catch(() => ({}));
                appendMessageToUI(
                    'assistant',
                    `⚠️ **Erro ao processar pergunta:** ${errData.detail || 'O serviço de IA não pôde responder agora. Verifique se o Ollama está ativo.'}`
                );
            }
        } catch (e) {
            clearInterval(timerInterval);
            const thinkingElem = document.getElementById('thinkingRow');
            if (thinkingElem) thinkingElem.remove();

            appendMessageToUI(
                'assistant',
                `❌ **Falha de comunicação com o servidor:** ${e.message}. Tente novamente.`
            );
        } finally {
            isGenerating = false;
            updateSendButtonState();
            chatInput.focus();
        }
    }

    async function renameConversation(id, currentTitle) {
        const newTitle = prompt('Digite o novo nome para esta conversa:', currentTitle);
        if (!newTitle || newTitle.trim() === '' || newTitle.trim() === currentTitle) return;

        try {
            const res = await fetch(`/conversation/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle.trim() })
            });

            if (res.ok) {
                if (currentConversationId === id) {
                    chatHeaderTitle.textContent = newTitle.trim();
                }
                await loadConversations();
            }
        } catch (e) {
            alert('Não foi possível renomear a conversa.');
        }
    }

    async function deleteConversation(id) {
        if (!confirm('Deseja realmente apagar esta conversa?')) return;

        try {
            const res = await fetch(`/conversation/${id}`, { method: 'DELETE' });
            if (res.ok) {
                if (currentConversationId === id) {
                    startNewChat();
                }
                await loadConversations();
            }
        } catch (e) {
            alert('Erro ao excluir conversa.');
        }
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Markdown Parser
    function formatMarkdown(text) {
        if (!text) return '';

        let out = escapeHtml(text);

        // Code blocks with syntax copy button
        out = out.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            return `
                <pre><button class="code-copy-btn">Copiar</button><code>${code.trim()}</code></pre>
            `;
        });

        // Inline code
        out = out.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers
        out = out.replace(/^### (.*$)/gim, '<h3 style="font-size: 16px; font-weight: 700; margin: 12px 0 6px; color: #93C5FD;">$1</h3>');
        out = out.replace(/^## (.*$)/gim, '<h2 style="font-size: 18px; font-weight: 700; margin: 14px 0 8px; color: #FFFFFF;">$1</h2>');
        out = out.replace(/^# (.*$)/gim, '<h1 style="font-size: 20px; font-weight: 800; margin: 16px 0 10px; color: #FFFFFF;">$1</h1>');

        // Bold
        out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Line breaks & paragraphs
        const paragraphs = out.split(/\n\n+/);
        out = paragraphs.map(p => {
            p = p.trim();
            if (p.startsWith('<pre>') || p.startsWith('<h1') || p.startsWith('<h2') || p.startsWith('<h3')) {
                return p;
            }
            // Lists
            if (p.includes('\n- ') || p.startsWith('- ')) {
                const items = p.split('\n- ');
                const listHtml = items.map((it, idx) => idx === 0 ? it.replace(/^- /, '') : it)
                    .filter(it => it.trim().length > 0)
                    .map(it => `<li>${it.replace(/\n/g, '<br>')}</li>`).join('');
                return `<ul>${listHtml}</ul>`;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');

        return out;
    }
});
