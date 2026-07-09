"""Fonte de dados ao vivo via MetaTrader 5.

Usa o pacote oficial `MetaTrader5` (Windows), que conversa com o terminal
MT5 instalado e logado na máquina do trader. O timeframe M2 é nativo do
MT5, então as barras de 2 minutos vêm prontas da plataforma — idênticas
às que o trader vê no gráfico, independente de zoom.
"""

from __future__ import annotations

import pandas as pd

from .base import BAR_COLUMNS, Bar, DataSource, validate_bars

try:  # importável apenas no Windows com o terminal instalado
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - ambiente de desenvolvimento Linux
    mt5 = None


class MT5Source(DataSource):
    def __init__(self, cfg: dict):
        if mt5 is None:
            raise RuntimeError(
                "Pacote MetaTrader5 indisponível. Ele só funciona no Windows, "
                "na máquina onde o terminal MT5 está instalado e logado."
            )
        self._cfg = cfg["mt5"]
        # nome lógico -> símbolo do broker
        self._symbol_map = {name: s["mt5_symbol"] for name, s in cfg["symbols"].items()}
        tf_min = cfg["bars"]["timeframe_minutes"]
        if tf_min != 2:
            raise ValueError("MVP fixado em barras de 2 minutos (M2)")
        self._timeframe = mt5.TIMEFRAME_M2
        self._last_closed: dict[str, pd.Timestamp] = {}

    def connect(self) -> None:
        kwargs = {}
        if self._cfg.get("terminal_path"):
            kwargs["path"] = self._cfg["terminal_path"]
        if self._cfg.get("login"):
            kwargs.update(
                login=int(self._cfg["login"]),
                password=self._cfg.get("password", ""),
                server=self._cfg.get("server", ""),
            )
        if not mt5.initialize(**kwargs):
            raise ConnectionError(f"mt5.initialize falhou: {mt5.last_error()}")
        for name, broker_symbol in self._symbol_map.items():
            if not mt5.symbol_select(broker_symbol, True):
                raise ValueError(
                    f"Símbolo '{broker_symbol}' (ativo {name}) não existe nesta corretora. "
                    "Confira o nome exato na Observação de Mercado e ajuste o config.yaml."
                )

    def close(self) -> None:
        mt5.shutdown()

    def _rates(self, symbol: str, count: int, start_pos: int = 0) -> pd.DataFrame:
        broker_symbol = self._symbol_map[symbol]
        rates = mt5.copy_rates_from_pos(broker_symbol, self._timeframe, start_pos, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})[BAR_COLUMNS]
        return validate_bars(df.astype(float))

    def history(self, symbol: str, n_bars: int) -> pd.DataFrame:
        # posição 0 é a barra em formação: pedimos a partir da 1 (só fechadas)
        df = self._rates(symbol, n_bars, start_pos=1)
        if not df.empty:
            # a última barra do histórico já foi consumida: o poll só deve
            # entregar barras fechadas DEPOIS dela
            self._last_closed[symbol] = df.index[-1]
        return df

    def poll_closed_bar(self, symbol: str) -> Bar | None:
        closed = self._rates(symbol, 1, start_pos=1)
        if closed.empty:
            return None
        ts = closed.index[-1]
        if self._last_closed.get(symbol) == ts:
            return None
        self._last_closed[symbol] = ts
        row = closed.iloc[-1]
        return Bar(symbol, ts.to_pydatetime(), row["open"], row["high"],
                   row["low"], row["close"], row["volume"])
