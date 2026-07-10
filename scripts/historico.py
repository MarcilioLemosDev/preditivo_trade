"""Histórico persistente das operações simuladas — para consultar depois.

    python scripts/historico.py                 # resumo de todos os ativos
    python scripts/historico.py --symbol BTC     # detalhe de um ativo
    python scripts/historico.py --symbol BTC --clips   # mostra cada pedaço

Lê os CSVs gravados em data/trades/ (persistentes entre execuções):
- {ATIVO}.csv        -> uma linha por OPERAÇÃO completa (P&L líquido)
- {ATIVO}_clips.csv  -> uma linha por PEDAÇO vendido (parcial/final)

Com --clips você confere se a soma do P&L dos pedaços de uma operação
bate com o P&L relatado daquela operação (coluna 'confere').
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


def _fmt_money(v: float) -> str:
    return f"US$ {v:+,.2f}"


def resumo_ativo(trades_path: Path) -> dict | None:
    if not trades_path.exists():
        return None
    df = pd.read_csv(trades_path)
    if df.empty:
        return None
    pnl = df["pnl_usd"].astype(float)
    n = len(df)
    wins = int((df["won"].astype(int) == 1).sum())
    return {
        "ativo": trades_path.stem,
        "operacoes": n,
        "winrate": wins / n,
        "ganhos": float(pnl[pnl >= 0].sum()),
        "perdas": float(pnl[pnl < 0].sum()),
        "liquido": float(pnl.sum()),
    }


def imprime_resumo(rows: list[dict], goal_trades: int, goal_winrate: float) -> None:
    if not rows:
        print("Nenhuma operação registrada ainda em data/trades/.")
        return
    print(f"{'ATIVO':<8} {'OPS':>5} {'WIN%':>6} {'GANHOS':>14} {'PERDAS':>14} "
          f"{'LÍQUIDO':>14}  META")
    print("-" * 78)
    for r in rows:
        if r["operacoes"] < goal_trades:
            meta = f"faltam {goal_trades - r['operacoes']} p/ avaliar"
        elif r["winrate"] >= goal_winrate:
            meta = f"OK >= {goal_winrate:.0%}"
        else:
            meta = f"ABAIXO de {goal_winrate:.0%}"
        print(f"{r['ativo']:<8} {r['operacoes']:>5} {r['winrate']:>6.0%} "
              f"{_fmt_money(r['ganhos']):>14} {_fmt_money(r['perdas']):>14} "
              f"{_fmt_money(r['liquido']):>14}  {meta}")


def detalhe_ativo(trades_path: Path, clips_path: Path, show_clips: bool) -> None:
    if not trades_path.exists():
        print(f"Sem operações para {trades_path.stem} ainda.")
        return
    trades = pd.read_csv(trades_path)
    print(f"\n=== {trades_path.stem}: {len(trades)} operação(ões) ===\n")
    clips = pd.read_csv(clips_path) if clips_path.exists() else pd.DataFrame()

    for _, op in trades.iterrows():
        resultado = "GANHO" if int(op["won"]) == 1 else "PERDA"
        print(f"#{int(op['op_id'])} {op['side']:<5} {op['entry_time']} -> {op['exit_time']} "
              f"| entrada {float(op['entry_price']):.4f} saída {float(op['exit_price']):.4f} "
              f"| capital US$ {float(op['size_usd']):,.0f} | {int(op['clips'])} pedaço(s) "
              f"| {resultado} {_fmt_money(float(op['pnl_usd']))}")
        if show_clips and not clips.empty:
            sub = clips[clips["op_id"] == op["op_id"]]
            soma = 0.0
            for _, c in sub.iterrows():
                soma += float(c["chunk_pnl"])
                print(f"     · pedaço {int(c['clip_no'])} [{c['reason']:<8}] "
                      f"US$ {float(c['chunk_usd']):>8,.0f} @ {float(c['price']):.4f} "
                      f"-> {_fmt_money(float(c['chunk_pnl']))} "
                      f"(resta US$ {float(c['remaining_usd']):,.0f})")
            confere = "OK" if abs(soma - float(op["pnl_usd"])) < 0.01 else "DIVERGE!"
            print(f"     soma dos pedaços = {_fmt_money(soma)}  [confere: {confere}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="detalha um ativo")
    parser.add_argument("--clips", action="store_true", help="mostra cada pedaço vendido")
    args = parser.parse_args()

    cfg = load_config()
    trades_dir = Path(cfg["storage"]["data_dir"]) / "trades"
    sim = cfg.get("simulation") or {}
    goal_trades = int(sim.get("goal_trades", 20))
    goal_winrate = float(sim.get("goal_winrate", 0.68))

    if not trades_dir.exists():
        print(f"Ainda não há histórico em {trades_dir}. Rode o app e deixe simular.")
        return

    if args.symbol:
        detalhe_ativo(trades_dir / f"{args.symbol}.csv",
                      trades_dir / f"{args.symbol}_clips.csv", args.clips)
        return

    rows = []
    for path in sorted(trades_dir.glob("*.csv")):
        if path.stem.endswith("_clips"):
            continue
        r = resumo_ativo(path)
        if r:
            rows.append(r)
    imprime_resumo(rows, goal_trades, goal_winrate)
    total_ops = sum(r["operacoes"] for r in rows)
    if total_ops:
        liquido = sum(r["liquido"] for r in rows)
        print("-" * 78)
        print(f"{'TOTAL':<8} {total_ops:>5} {'':>6} {'':>14} {'':>14} "
              f"{_fmt_money(liquido):>14}")
    print("\nDetalhe de um ativo:  python scripts/historico.py --symbol BTC --clips")


if __name__ == "__main__":
    main()
