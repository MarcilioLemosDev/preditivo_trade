from src.signals import COMPRA, FORA, MANTER, SAIR, VENDA, decide

CFG = {"buy_prob": 0.60, "margin": 0.25, "exit_prob": 0.55}


def _p(alta, lateral, baixa):
    return {"ALTA": alta, "LATERAL": lateral, "BAIXA": baixa}


def test_buy_when_edge_is_clear():
    assert decide(_p(0.65, 0.20, 0.15), None, CFG).action == COMPRA


def test_no_signal_without_margin():
    # prob alta o bastante, mas vantagem sobre BAIXA menor que a margem
    assert decide(_p(0.60, 0.02, 0.38), None, CFG).action == FORA


def test_no_signal_when_lateral_dominates():
    assert decide(_p(0.20, 0.60, 0.20), None, CFG).action == FORA


def test_sell_symmetric():
    assert decide(_p(0.10, 0.20, 0.70), None, CFG).action == VENDA


def test_exit_long_on_opposite_pressure():
    assert decide(_p(0.20, 0.20, 0.60), "COMPRADO", CFG).action == SAIR


def test_hold_long_when_no_pressure():
    assert decide(_p(0.40, 0.40, 0.20), "COMPRADO", CFG).action == MANTER


def test_exit_short_on_opposite_pressure():
    assert decide(_p(0.60, 0.20, 0.20), "VENDIDO", CFG).action == SAIR
