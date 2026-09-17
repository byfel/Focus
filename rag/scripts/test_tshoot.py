import sys

from rag.tshoot import run_tshoot, TShootError


QUERY = """
O nome do dispositivo aparece em texto vermelho no Dante Controller
e não aceita roteamento.
""".strip()


# Contexto inicial de teste.
# Depois vamos substituir por trechos reais recuperados do RAG.
CONTEXT = """
DOCUMENTO: AUD-MAN-DanteController-4.18.x-v1.0.pdf

--- FONTE 1 ---
Página: TESTE
Conteúdo:
[COLOCAR AQUI O TRECHO DO MANUAL RELACIONADO AO DISPOSITIVO
EM VERMELHO E AO PROBLEMA DE ROTEAMENTO]
"""


EVIDENCE_AUDIT = """
FATOS_DOCUMENTADOS:
- O contexto disponível informa que determinado dispositivo
  pode aparecer em vermelho no Dante Controller.
- O contexto informa que existe uma relação entre o estado
  apresentado no Dante Controller e o roteamento.

CONCLUSOES_DIRETAS:
- O estado visual apresentado pelo dispositivo deve ser
  investigado antes de concluir que existe uma falha de roteamento.

INFERENCIAS_NAO_COMPROVADAS:
- Não há evidência suficiente no contexto para afirmar
  qual é a causa específica do problema.
"""


def main():
    print("=" * 70)
    print("T-SHOOT — TESTE ISOLADO")
    print("=" * 70)

    print("\nPERGUNTA:")
    print(QUERY)

    print("\nExecutando análise...\n")

    try:
        result = run_tshoot(
            query=QUERY,
            context=CONTEXT,
            evidence_audit=EVIDENCE_AUDIT,
        )

    except TShootError as exc:
        print("ERRO:")
        print(exc)
        sys.exit(1)

    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print()
    print(result)


if __name__ == "__main__":
    main()