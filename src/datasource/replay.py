"""Fonte de replay: reproduz histórico gravado como se fosse ao vivo.

Serve para desenvolver e testar todo o pipeline (features, modelo,
sinais, UI) sem depender do terminal MT5 — inclusive em Linux/CI —
e para simular pregões inteiros em velocidade acelerada.
"""

from __future__ import annotations

import pandas as pd

from .base import BAR_COLUMNS, Bar, DataSource, validate_bars


class ReplaySource(DataSource):
    def __init__(self, frames: dict[str, pd.DataFrame], warmup_bars: int = 0):
        """frames: {nome_do_ativo: DataFrame OHLCV completo}.

        As primeiras `warmup_bars` barras saem via history(); o restante é
        entregue uma a uma por poll_closed_bar(), na ordem do índice.
        """
        self._frames = {sym: validate_bars(df[BAR_COLUMNS].copy()) for sym, df in frames.items()}
        self._cursor = {sym: warmup_bars for sym in frames}
        self._warmup = warmup_bars

    @classmethod
    def from_csv(cls, paths: dict[str, str], warmup_bars: int = 0) -> "ReplaySource":
        frames = {
            sym: pd.read_csv(path, index_col="time", parse_dates=["time"])
            for sym, path in paths.items()
        }
        return cls(frames, warmup_bars)

    def connect(self) -> None:  # nada a fazer
        pass

    def close(self) -> None:
        pass

    def history(self, symbol: str, n_bars: int) -> pd.DataFrame:
        return self._frames[symbol].iloc[: self._warmup].tail(n_bars)

    def poll_closed_bar(self, symbol: str) -> Bar | None:
        df = self._frames[symbol]
        i = self._cursor[symbol]
        if i >= len(df):
            return None
        self._cursor[symbol] = i + 1
        row = df.iloc[i]
        ts = df.index[i]
        return Bar(symbol, ts.to_pydatetime(), row["open"], row["high"],
                   row["low"], row["close"], row["volume"])

    def exhausted(self, symbol: str) -> bool:
        return self._cursor[symbol] >= len(self._frames[symbol])
