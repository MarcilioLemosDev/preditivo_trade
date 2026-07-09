"""Engenharia de features sobre barras de 2 minutos.

REGRA DE OURO: a feature da barra t só pode usar informação disponível
no fechamento de t (a própria barra t e anteriores). Nada de shift(-1),
nada de estatísticas do dataset inteiro. Todos os cálculos são rolling/
ewm, então o valor em t é idêntico ao que seria calculado ao vivo — o
teste de "no lookahead" em tests/test_features.py garante isso.

A visão MICRO (janelas curtas) captura o momento; a visão MACRO
(janelas longas) captura o contexto — o equivalente ao trader tirar o
zoom do gráfico, sem depender de zoom nenhum.

Indicadores implementados manualmente (RSI, EMA, ATR, Bollinger) para
não depender de bibliotecas de TA com histórico de quebra entre versões.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(
    df: pd.DataFrame,
    micro_windows: list[int] = (3, 5, 10),
    macro_windows: list[int] = (30, 60, 120),
) -> pd.DataFrame:
    """df: OHLCV indexado por tempo. Retorna features alinhadas ao índice."""
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    ret1 = close.pct_change()

    # --- anatomia do candle atual (micro extremo) ---
    rng = (high - low).replace(0.0, np.nan)
    body = df["close"] - df["open"]
    out["body_pct"] = body / close
    out["body_ratio"] = body.abs() / rng
    out["upper_wick"] = (high - df[["open", "close"]].max(axis=1)) / rng
    out["lower_wick"] = (df[["open", "close"]].min(axis=1) - low) / rng
    out["ret_1"] = ret1

    # sequência de cores: +n = n barras verdes seguidas, -n = vermelhas
    color = np.sign(body).replace(0.0, np.nan).ffill().fillna(0.0)
    blocks = (color != color.shift()).cumsum()
    out["streak"] = color * color.groupby(blocks).cumcount().add(1)

    # --- janelas micro e macro ---
    for w in list(micro_windows) + list(macro_windows):
        out[f"ret_{w}"] = close.pct_change(w)
        out[f"vol_{w}"] = ret1.rolling(w).std()
        wrange = (high.rolling(w).max() - low.rolling(w).min()).replace(0.0, np.nan)
        out[f"pos_range_{w}"] = (close - low.rolling(w).min()) / wrange

    # volume relativo ao contexto macro
    w_macro = max(macro_windows)
    vol_ma = vol.rolling(w_macro).mean().replace(0.0, np.nan)
    out["volume_rel"] = vol / vol_ma

    # --- indicadores clássicos ---
    out["rsi_14"] = _rsi(close)
    for span in (9, 21):
        out[f"ema{span}_dist"] = close / close.ewm(span=span, adjust=False).mean() - 1
    out["atr_14_pct"] = _atr(df) / close
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0.0, np.nan)
    out["bb_z"] = (close - sma20) / (2 * std20)

    # --- hora do dia (sessões têm dinâmicas próprias) ---
    minutes = df.index.hour * 60 + df.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)

    return out.replace([np.inf, -np.inf], np.nan)


def warmup_bars(macro_windows: list[int] = (30, 60, 120)) -> int:
    """Barras mínimas antes de as features macro ficarem completas."""
    return max(macro_windows) + 5
