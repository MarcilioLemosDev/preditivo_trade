"""Rotulagem das barras nas classes ALTA / LATERAL / BAIXA.

O rótulo da barra t descreve o retorno do fechamento de t+1 contra o
fechamento de t (horizonte fixo de 1 barra = 2 minutos):

    r_next > +theta  -> ALTA (2)
    r_next < -theta  -> BAIXA (0)
    caso contrário   -> LATERAL (1)

theta é adaptativo por ativo (modo quantile): o quantil dos |retornos|
PASSADOS numa janela móvel — assim BTC e MSFT, com volatilidades de
mundos diferentes, produzem classes igualmente balanceadas.

Barras seguidas de gap de sessão (a próxima barra não é exatamente
t + 2min: fim de pregão, feriado, buraco de feed) recebem rótulo NaN e
saem do treino — o "próximo movimento" delas não é um movimento de
2 minutos, é um pulo de horas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "BAIXA", 1: "LATERAL", 2: "ALTA"}


def make_labels(df: pd.DataFrame, cfg_labeling: dict, timeframe_minutes: int = 2) -> pd.DataFrame:
    """df: OHLCV indexado por tempo. Retorna [r_next, theta, label]."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["r_next"] = close.shift(-1) / close - 1

    # gap de sessão: o rótulo só vale se a próxima barra for contígua
    next_time = df.index.to_series().shift(-1)
    contiguous = (next_time - df.index.to_series()) == pd.Timedelta(minutes=timeframe_minutes)

    mode = cfg_labeling["mode"]
    if mode == "fixed":
        out["theta"] = float(cfg_labeling["fixed_theta_pct"])
    elif mode == "quantile":
        window = int(cfg_labeling["window"])
        q = float(cfg_labeling["lateral_quantile"])
        abs_ret = close.pct_change().abs()  # retorno passado: conhecido em t
        out["theta"] = abs_ret.rolling(window, min_periods=max(50, window // 5)).quantile(q)
    else:
        raise ValueError(f"labeling.mode desconhecido: {mode}")

    label = pd.Series(1.0, index=df.index)
    label[out["r_next"] > out["theta"]] = 2.0
    label[out["r_next"] < -out["theta"]] = 0.0
    label[out["r_next"].isna() | out["theta"].isna() | ~contiguous] = np.nan
    out["label"] = label
    return out
