"""O teste mais importante do projeto: garantia de não-vazamento.

Se a feature da barra t mudar quando barras futuras são adicionadas ao
DataFrame, há look-ahead — e qualquer métrica de modelo vira mentira.
"""

import numpy as np

from src.features import build_features, warmup_bars


def test_no_lookahead(ohlcv):
    """Feature em t calculada com df[:t] deve ser idêntica à calculada com o df inteiro."""
    full = build_features(ohlcv)
    for cut in (300, 600, 900):
        prefix = build_features(ohlcv.iloc[:cut])
        row_full = full.iloc[cut - 1]
        row_prefix = prefix.iloc[-1]
        np.testing.assert_allclose(
            row_prefix.to_numpy(dtype=float),
            row_full.to_numpy(dtype=float),
            rtol=1e-9,
            err_msg=f"look-ahead detectado na barra {cut - 1}",
        )


def test_alignment_and_sanity(ohlcv):
    feats = build_features(ohlcv)
    assert feats.index.equals(ohlcv.index)
    assert not np.isinf(feats.to_numpy(dtype=float)).any()
    # após o warmup, nenhuma feature deve ser NaN
    warm = feats.iloc[warmup_bars():]
    assert not warm.isna().any().any(), warm.isna().sum()[lambda s: s > 0]


def test_streak_signal(ohlcv):
    feats = build_features(ohlcv)
    body = ohlcv["close"] - ohlcv["open"]
    # barra 3 verdes seguidas -> streak >= 3
    greens = (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0)
    assert (feats.loc[greens, "streak"] >= 3).all()
