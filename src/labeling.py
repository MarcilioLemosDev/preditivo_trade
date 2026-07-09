"""Rotulagem das barras nas classes ALTA / LATERAL / BAIXA.

O rótulo da barra t descreve o retorno do fechamento de t+h contra o
fechamento de t, onde h é o HORIZONTE em barras escolhido pelo trader
(config prediction.horizon_bars: 1 = 2 min à frente, 2 = 4 min,
3 = 6 min...):

    r > +theta  -> ALTA (2)
    r < -theta  -> BAIXA (0)
    caso contrário -> LATERAL (1)

theta é adaptativo por ativo E por horizonte (modo quantile): o quantil
dos |retornos de h barras| PASSADOS numa janela móvel — assim BTC e
MSFT, com volatilidades de mundos diferentes, produzem classes
igualmente balanceadas, e ao aumentar o horizonte o theta cresce junto
com a volatilidade acumulada.

Barras cujo alvo cruza gap de sessão (t+h não está exatamente h barras
de 2 min à frente: fim de pregão, feriado, buraco de feed) recebem
rótulo NaN e saem do treino — o movimento delas não é um movimento de
h*2 minutos, é um pulo de horas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "BAIXA", 1: "LATERAL", 2: "ALTA"}


def make_labels(df: pd.DataFrame, cfg_labeling: dict, timeframe_minutes: int = 2,
                horizon_bars: int = 1) -> pd.DataFrame:
    """df: OHLCV indexado por tempo. Retorna [r_next, theta, label]."""
    h = int(horizon_bars)
    if h < 1:
        raise ValueError("horizon_bars deve ser >= 1")
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["r_next"] = close.shift(-h) / close - 1

    # o rótulo só vale se a barra-alvo estiver exatamente h barras à frente
    target_time = df.index.to_series().shift(-h)
    contiguous = (target_time - df.index.to_series()) == pd.Timedelta(minutes=h * timeframe_minutes)

    mode = cfg_labeling["mode"]
    if mode == "fixed":
        out["theta"] = float(cfg_labeling["fixed_theta_pct"])
    elif mode == "quantile":
        window = int(cfg_labeling["window"])
        q = float(cfg_labeling["lateral_quantile"])
        abs_ret = close.pct_change(h).abs()  # retorno passado de h barras: conhecido em t
        out["theta"] = abs_ret.rolling(window, min_periods=max(50, window // 5)).quantile(q)
    else:
        raise ValueError(f"labeling.mode desconhecido: {mode}")

    label = pd.Series(1.0, index=df.index)
    label[out["r_next"] > out["theta"]] = 2.0
    label[out["r_next"] < -out["theta"]] = 0.0
    label[out["r_next"].isna() | out["theta"].isna() | ~contiguous] = np.nan
    out["label"] = label
    return out
