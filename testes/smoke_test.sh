#!/usr/bin/env bash
#
# smoke_test.sh — teste de fumaça end-to-end contra a API do RAG
# rodando de verdade (precisa do Qdrant e do Ollama no ar).
#
# Uso:
#   BASE_URL=http://localhost:8000 ./smoke_test.sh
#
# Variáveis de ambiente aceitas:
#   BASE_URL             (default: http://localhost:8000)
#   RAG_TEST_USERNAME     (default: admin)
#   RAG_TEST_PASSWORD     (default: admin123)
#
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
USERNAME="${RAG_TEST_USERNAME:-admin}"
PASSWORD="${RAG_TEST_PASSWORD:-admin123}"

COOKIE_JAR="$(mktemp)"
trap 'rm -f "$COOKIE_JAR"' EXIT

pass() { echo "  OK: $1"; }
fail() { echo "  FALHOU: $1"; exit 1; }

echo "Testando contra: ${BASE_URL}"
echo

echo "== 1) /health =="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/health")
[[ "$STATUS" == "200" ]] && pass "health respondeu 200" || fail "health respondeu ${STATUS}"

echo "== 2) /me sem login (deve dar 401) =="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIE_JAR" -b "$COOKIE_JAR" "${BASE_URL}/me")
[[ "$STATUS" == "401" ]] && pass "/me sem sessão retornou 401" || fail "/me sem sessão retornou ${STATUS} (esperado 401)"

echo "== 3) /login com credenciais válidas =="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -X POST "${BASE_URL}/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${USERNAME}\", \"password\": \"${PASSWORD}\"}")
[[ "$STATUS" == "200" ]] && pass "login OK" || fail "login retornou ${STATUS}"

echo "== 4) /me com sessão válida =="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIE_JAR" -b "$COOKIE_JAR" "${BASE_URL}/me")
[[ "$STATUS" == "200" ]] && pass "/me com sessão retornou 200" || fail "/me com sessão retornou ${STATUS}"

echo "== 5) /models lista os modelos configurados =="
curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" "${BASE_URL}/models"
echo
echo

echo "== 6) /ask com modelo inválido (deve dar 400, sem chegar a chamar o Ollama) =="
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -X POST "${BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "teste", "model": "modelo-que-nao-existe"}')
[[ "$STATUS" == "400" ]] && pass "modelo inválido retornou 400" || fail "modelo inválido retornou ${STATUS}"

echo "== 7) /ask com pergunta real (pode demorar bastante em hardware Maxwell) =="
time curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -X POST "${BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "Como faço para excluir um projeto no Flame?"}' \
    | python3 -m json.tool
echo

echo "== 8) /logout invalida a sessão de verdade (era o bug corrigido) =="
curl -s -o /dev/null -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "${BASE_URL}/logout"
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -c "$COOKIE_JAR" -b "$COOKIE_JAR" "${BASE_URL}/me")
if [[ "$STATUS" == "401" ]]; then
    pass "/me após logout retornou 401 (sessão realmente invalidada no servidor)"
else
    fail "/me após logout retornou ${STATUS} (esperado 401 — logout NÃO está invalidando a sessão!)"
fi

echo
echo "Todos os testes de fumaça passaram."
