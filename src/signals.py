"""Conversão de probabilidades em sugestões acionáveis.

O modelo entrega P(ALTA), P(LATERAL), P(BAIXA); aqui entram as regras de
negócio: só sugerir quando há vantagem clara, e tratar "ficar de fora"
como o sinal mais comum e legítimo. Quem executa é sempre o humano —
este módulo apenas sugere e explica.
"""

from __future__ import annotations

from dataclasses import dataclass

# ações possíveis
COMPRA, VENDA, SAIR, MANTER, FORA = "COMPRA", "VENDA", "SAIR", "MANTER", "FORA"


@dataclass(frozen=True)
class Signal:
    action: str
    reason: str


def decide(probs: dict[str, float], position: str | None, cfg_signals: dict) -> Signal:
    """probs: {"ALTA": p, "LATERAL": p, "BAIXA": p}. position: None|COMPRADO|VENDIDO."""
    p_up, p_down = probs["ALTA"], probs["BAIXA"]
    buy_prob = float(cfg_signals["buy_prob"])
    margin = float(cfg_signals["margin"])
    exit_prob = float(cfg_signals["exit_prob"])

    if position == "COMPRADO":
        if p_down >= exit_prob:
            return Signal(SAIR, f"P(BAIXA)={p_down:.0%} >= {exit_prob:.0%} contra a posição comprada")
        return Signal(MANTER, f"P(BAIXA)={p_down:.0%} abaixo do gatilho de saída")

    if position == "VENDIDO":
        if p_up >= exit_prob:
            return Signal(SAIR, f"P(ALTA)={p_up:.0%} >= {exit_prob:.0%} contra a posição vendida")
        return Signal(MANTER, f"P(ALTA)={p_up:.0%} abaixo do gatilho de saída")

    if p_up >= buy_prob and (p_up - p_down) >= margin:
        return Signal(COMPRA, f"P(ALTA)={p_up:.0%} com vantagem de {p_up - p_down:.0%} sobre BAIXA")
    if p_down >= buy_prob and (p_down - p_up) >= margin:
        return Signal(VENDA, f"P(BAIXA)={p_down:.0%} com vantagem de {p_down - p_up:.0%} sobre ALTA")
    return Signal(FORA, "sem vantagem estatística suficiente")
