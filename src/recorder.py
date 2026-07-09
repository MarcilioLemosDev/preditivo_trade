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


class Recorder:
    def __init__(self, data_dir: str | Path):
        self.bars_dir = Path(data_dir) / "bars"
        self.signals_dir = Path(data_dir) / "signals"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)

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

    def load_bars(self, symbol: str) -> pd.DataFrame:
        path = self.bars_dir / f"{symbol}.csv"
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.read_csv(path, index_col="time", parse_dates=["time"])
        return df[~df.index.duplicated(keep="last")].sort_index()
