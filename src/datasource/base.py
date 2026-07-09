"""Contrato único de fonte de dados.

Toda fonte (MT5 ao vivo, replay de histórico, futura captura de tela)
entrega o mesmo formato: DataFrame OHLCV indexado por horário UTC do
INÍCIO da barra, com colunas [open, high, low, close, volume]. Somente
barras FECHADAS são entregues — a barra em formação nunca sai daqui,
para o resto do pipeline não ter como vazar o futuro.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Bar:
    symbol: str
    time: datetime  # início da barra, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_row(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.open, self.high, self.low, self.close, self.volume]],
            columns=BAR_COLUMNS,
            index=pd.DatetimeIndex([self.time], name="time"),
        )


class DataSource(ABC):
    """Interface de fonte de dados de barras fechadas."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def history(self, symbol: str, n_bars: int) -> pd.DataFrame:
        """Últimas n_bars barras FECHADAS (index UTC, colunas BAR_COLUMNS)."""

    @abstractmethod
    def poll_closed_bar(self, symbol: str) -> Bar | None:
        """Retorna a barra recém-fechada desde a última chamada, ou None.

        Idempotente entre fechamentos: chamadas repetidas dentro da mesma
        janela de 2 minutos retornam None até uma nova barra fechar.
        """


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Sanidade básica: colunas, ordenação, duplicatas e OHLC coerente."""
    if list(df.columns) != BAR_COLUMNS:
        raise ValueError(f"colunas esperadas {BAR_COLUMNS}, recebidas {list(df.columns)}")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    return df[~bad]
