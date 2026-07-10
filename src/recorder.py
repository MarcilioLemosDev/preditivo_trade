"""Gravação em disco de tudo que acontece: barras e sinais.

CSV com append linha a linha — trivial de auditar, resistente a queda
do programa no meio do pregão, e suficiente para o volume de barras de
2 minutos. Esses arquivos alimentam o retreino e o backtest.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .datasource.base import BAR_COLUMNS, Bar

BARS_HEADER = ["time", *BAR_COLUMNS]
SIGNALS_HEADER = ["time", "symbol", "close", "p_baixa", "p_lateral", "p_alta", "action", "reason"]
TRADES_HEADER = ["op_id", "symbol", "side", "entry_time", "exit_time", "entry_price",
                 "exit_price", "size_usd", "pnl_usd", "won", "clips", "reason"]
CLIPS_HEADER = ["op_id", "symbol", "clip_no", "side", "time", "entry_price", "price",
                "chunk_usd", "chunk_pnl", "remaining_usd", "reason"]


class Recorder:
    def __init__(self, data_dir: str | Path):
        self.bars_dir = Path(data_dir) / "bars"
        self.signals_dir = Path(data_dir) / "signals"
        self.trades_dir = Path(data_dir) / "trades"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, header: list[str], row: list) -> None:
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)

    def record_bar(self, bar: Bar) -> None:
        self._append(
            self.bars_dir / f"{bar.symbol}.csv", BARS_HEADER,
            [bar.time.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume],
        )

    def record_signal(self, symbol: str, time, close: float,
                      probs: dict[str, float], action: str, reason: str) -> None:
        self._append(
            self.signals_dir / f"{symbol}.csv", SIGNALS_HEADER,
            [time.isoformat(), symbol, close,
             f"{probs['BAIXA']:.4f}", f"{probs['LATERAL']:.4f}", f"{probs['ALTA']:.4f}",
             action, reason],
        )

    def record_trade(self, trade) -> None:
        self._append(
            self.trades_dir / f"{trade.symbol}.csv", TRADES_HEADER,
            [trade.op_id, trade.symbol, trade.side,
             trade.entry_time.isoformat(), trade.exit_time.isoformat(),
             f"{trade.entry_price:.6f}", f"{trade.exit_price:.6f}",
             f"{trade.size_usd:.2f}", f"{trade.pnl_usd:.2f}", int(trade.won),
             trade.clips, trade.reason],
        )

    def record_clip(self, clip) -> None:
        self._append(
            self.trades_dir / f"{clip.symbol}_clips.csv", CLIPS_HEADER,
            [clip.op_id, clip.symbol, clip.clip_no, clip.side, clip.time.isoformat(),
             f"{clip.entry_price:.6f}", f"{clip.price:.6f}", f"{clip.chunk_usd:.2f}",
             f"{clip.chunk_pnl:.2f}", f"{clip.remaining_usd:.2f}", clip.reason],
        )

    def trade_summary(self, symbol: str) -> dict:
        """Resumo das operações já gravadas, para semear o placar cumulativo."""
        path = self.trades_dir / f"{symbol}.csv"
        if not path.exists():
            return {"n_trades": 0, "wins": 0, "gross_profit": 0.0,
                    "gross_loss": 0.0, "next_op_id": 0}
        df = pd.read_csv(path)
        if df.empty:
            return {"n_trades": 0, "wins": 0, "gross_profit": 0.0,
                    "gross_loss": 0.0, "next_op_id": 0}
        pnl = df["pnl_usd"].astype(float)
        return {
            "n_trades": int(len(df)),
            "wins": int((df["won"].astype(int) == 1).sum()),
            "gross_profit": float(pnl[pnl >= 0].sum()),
            "gross_loss": float(pnl[pnl < 0].sum()),
            "next_op_id": int(df["op_id"].astype(int).max()) + 1,
        }

    def load_bars(self, symbol: str) -> pd.DataFrame:
        path = self.bars_dir / f"{symbol}.csv"
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.read_csv(path, index_col="time", parse_dates=["time"])
        return df[~df.index.duplicated(keep="last")].sort_index()
