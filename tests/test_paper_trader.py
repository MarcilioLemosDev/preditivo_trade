"""Testes do simulador de operações (paper trading)."""

from datetime import datetime, timedelta, timezone

from src.paper_trader import LONG, SHORT, PaperTrader, Seed

CFG = {
    "prediction": {"horizon_bars": 3},
    "bars": {"timeframe_minutes": 2},
    "simulation": {
        "entry_prob": 0.60, "entry_margin": 0.20,
        "partial_exit_prob": 0.45, "partial_fraction": 0.5, "exit_prob": 0.55,
        "min_hold_seconds": 40, "window_seconds": 600,
        "max_entries_per_window": 2, "max_exits_per_window": 2,
        "max_hold_minutes": 10, "size_min_usd": 2000, "size_max_usd": 4000,
        "cost_pct": 0.0, "goal_trades": 20, "goal_winrate": 0.68,
    },
}

T0 = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)


def _bar(pt, minutes, price, alta, baixa):
    lateral = max(0.0, 1 - alta - baixa)
    return pt.on_bar(T0 + timedelta(minutes=minutes),
                     price, {"ALTA": alta, "LATERAL": lateral, "BAIXA": baixa})


def test_max_hold_is_five_bars_for_ten_minutes():
    pt = PaperTrader("X", CFG)
    assert pt.max_hold_bars == 5  # 10 min / 2 min


def test_enters_long_and_takes_profit_on_reversal():
    pt = PaperTrader("X", CFG)
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)          # entra LONG
    assert pt.pos is not None and pt.pos.side == LONG
    stats = _bar(pt, 2, 101.0, alta=0.10, baixa=0.80)  # reversão forte -> zera com lucro
    assert stats["sim_trades"] == 1
    assert stats["sim_wins"] == 1
    assert stats["sim_pnl"] > 0
    assert pt.pos is None


def test_short_profits_when_price_falls():
    pt = PaperTrader("X", CFG)
    _bar(pt, 0, 100.0, alta=0.10, baixa=0.80)          # entra SHORT
    assert pt.pos.side == SHORT
    stats = _bar(pt, 2, 98.0, alta=0.80, baixa=0.10)   # preço caiu, reversão -> lucro no short
    assert stats["sim_trades"] == 1 and stats["sim_pnl"] > 0


def test_min_hold_40s_blocks_immediate_exit():
    """Barras separadas por 30s (< 40s) não podem fechar a posição."""
    pt = PaperTrader("X", CFG)
    t = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    pt.on_bar(t, 100.0, {"ALTA": 0.80, "LATERAL": 0.10, "BAIXA": 0.10})
    stats = pt.on_bar(t + timedelta(seconds=30), 101.0,
                      {"ALTA": 0.10, "LATERAL": 0.10, "BAIXA": 0.80})
    assert stats["sim_trades"] == 0        # não fechou: hold < 40s
    assert pt.pos is not None


def test_partial_exit_then_full_is_one_operation():
    pt = PaperTrader("X", CFG)
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)          # entra LONG, size cheio
    size0 = pt.pos.initial_size
    _bar(pt, 2, 101.0, alta=0.40, baixa=0.48)          # pressão moderada -> parcial (50%)
    assert pt.pos is not None
    assert abs(pt.pos.remaining - size0 / 2) < 1e-6
    assert pt.pos.clips == 1
    stats = _bar(pt, 4, 102.0, alta=0.10, baixa=0.80)  # pressão forte -> zera o resto
    assert stats["sim_trades"] == 1                    # UMA operação, com 2 pedaços
    assert pt.last_closed.clips == 2
    assert stats["sim_pnl"] > 0


def test_exits_capped_at_two_per_window():
    """Teto de 2 saídas por janela de 10 min: a 3ª fica bloqueada."""
    pt = PaperTrader("X", CFG)
    # entra e faz parciais repetidas com pressão moderada constante
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)
    _bar(pt, 2, 100.5, alta=0.40, baixa=0.48)   # parcial 1
    _bar(pt, 4, 100.6, alta=0.40, baixa=0.48)   # parcial 2
    clips_after_two = pt.pos.clips
    _bar(pt, 6, 100.7, alta=0.40, baixa=0.48)   # BLOQUEADA (já houve 2 saídas na janela)
    assert pt.pos.clips == clips_after_two == 2


def test_position_size_within_configured_band():
    pt = PaperTrader("X", CFG)
    _bar(pt, 0, 100.0, alta=0.99, baixa=0.005)   # confiança máxima -> perto do teto
    assert 2000 <= pt.pos.initial_size <= 4000


def test_goal_ok_requires_20_trades_and_68pct():
    pt = PaperTrader("X", CFG, seed=Seed(n_trades=19, wins=19, next_op_id=19))
    assert pt.stats()["sim_goal_reached"] is False   # ainda 19
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)
    stats = _bar(pt, 2, 101.0, alta=0.10, baixa=0.80)  # 20ª operação, vitória
    assert stats["sim_trades"] == 20
    assert stats["sim_goal_reached"] is True
    assert stats["sim_goal_ok"] is True                # 20/20 = 100% >= 68%


def test_seed_makes_scoreboard_cumulative():
    pt = PaperTrader("X", CFG, seed=Seed(n_trades=5, wins=4, gross_profit=100.0,
                                         gross_loss=-20.0, next_op_id=5))
    s = pt.stats()
    assert s["sim_trades"] == 5 and s["sim_wins"] == 4
    assert abs(s["sim_pnl"] - 80.0) < 1e-9
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)   # nova operação usa op_id 5
    assert pt.pos.op_id == 5


def test_clips_reconcile_with_operation_pnl():
    """A soma do P&L dos pedaços tem de bater com o P&L da operação."""
    pt = PaperTrader("X", CFG)
    _bar(pt, 0, 100.0, alta=0.80, baixa=0.10)
    all_clips = []
    all_clips += _collect_clips(pt)
    _bar(pt, 2, 101.0, alta=0.40, baixa=0.48)   # parcial
    all_clips += _collect_clips(pt)
    _bar(pt, 4, 102.0, alta=0.10, baixa=0.80)   # zera
    all_clips += _collect_clips(pt)
    soma = sum(c.chunk_pnl for c in all_clips)
    assert abs(soma - pt.last_closed.pnl_usd) < 1e-6


def _collect_clips(pt):
    return list(pt.last_clips)
