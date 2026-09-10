"""
Testes da lógica de RRF (Reciprocal Rank Fusion) do retriever.py.
Testa só `calculate_rrf_score`, que é pura — não toca Qdrant nem
Ollama.

Ajuste o import conforme o caminho real (`rag.rag.retriever` ou
o que corresponder no seu projeto).

Rodar com:
    pytest tests/test_retriever_logic.py -v
"""
import pytest

from rag import retriever  # <-- ajuste este import se necessário


def test_rrf_score_single_rank():
    # RRF_K=60 -> 1 / (60 + 1)
    assert retriever.calculate_rrf_score([1]) == pytest.approx(1 / 61)


def test_rrf_score_sums_across_multiple_appearances():
    ranks = [1, 3]
    expected = 1 / 61 + 1 / 63
    assert retriever.calculate_rrf_score(ranks) == pytest.approx(expected)


def test_rrf_score_better_rank_scores_higher():
    assert retriever.calculate_rrf_score([1]) > retriever.calculate_rrf_score([10])


def test_rrf_score_more_appearances_scores_higher():
    """Um chunk que aparece em duas consultas expandidas (mesmo em
    posições piores) deve tender a pontuar mais que um que aparece
    uma vez só — é a lógica que sustenta a Query Expansion."""
    one_appearance = retriever.calculate_rrf_score([5])
    two_appearances = retriever.calculate_rrf_score([5, 8])
    assert two_appearances > one_appearance


def test_rrf_score_empty_ranks_is_zero():
    assert retriever.calculate_rrf_score([]) == 0
