"""
Testes de `_build_context` do generator.py — o truncamento de
contexto por MAX_CONTEXT_CHARS. Não faz chamada ao Ollama.

Ajuste o import conforme o caminho real (`rag.rag.generator`).

Rodar com:
    pytest tests/test_generator_logic.py -v
"""
from rag import generator  # <-- ajuste este import se necessário


def _fake_result(text, score=0.9, document="doc.pdf", page=1, chunk=1):
    return {
        "score": score,
        "chunk": {
            "document": document,
            "page": page,
            "chunk": chunk,
            "text": text,
        },
    }


def test_build_context_includes_all_when_within_budget(monkeypatch):
    monkeypatch.setattr(generator, "MAX_CONTEXT_CHARS", 10_000)
    results = [_fake_result("conteúdo curto") for _ in range(3)]

    context = generator._build_context(results)

    assert context.count("FONTE") == 3


def test_build_context_truncates_when_over_budget(monkeypatch, caplog):
    monkeypatch.setattr(generator, "MAX_CONTEXT_CHARS", 300)
    results = [_fake_result("x" * 200) for _ in range(5)]

    with caplog.at_level("WARNING"):
        context = generator._build_context(results)

    assert context.count("FONTE") < 5
    assert "truncado" in caplog.text.lower()


def test_build_context_empty_results():
    assert generator._build_context([]) == ""


def test_build_context_missing_fields_use_defaults():
    """Chunk sem 'document'/'page'/'chunk' não deve quebrar, só
    usar os defaults ('desconhecido', '?', '?')."""
    result = {"score": 0.5, "chunk": {"text": "algum texto"}}

    context = generator._build_context([result])

    assert "desconhecido" in context
    assert "algum texto" in context
