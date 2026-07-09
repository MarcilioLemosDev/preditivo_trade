import numpy as np
import pandas as pd

from src.labeling import make_labels

CFG_FIXED = {"mode": "fixed", "fixed_theta_pct": 0.001, "window": 500, "lateral_quantile": 0.33}
CFG_QUANTILE = {"mode": "quantile", "lateral_quantile": 0.33, "window": 300, "fixed_theta_pct": 0.0005}


def _tiny_frame():
    """5 barras contíguas de 2 min com retornos conhecidos."""
    index = pd.date_range("2026-06-01 14:00", periods=5, freq="2min", tz="UTC", name="time")
    close = pd.Series([100.0, 100.5, 100.4, 100.4, 99.0], index=index)  # +0.5%, -0.1%, 0%, -1.4%
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1.0}
    )


def test_fixed_labels_deterministic():
    labels = make_labels(_tiny_frame(), CFG_FIXED)
    # theta = 0.1%: +0.5% -> ALTA(2); -0.0995% -> LATERAL(1); 0% -> LATERAL(1); -1.39% -> BAIXA(0)
    assert list(labels["label"].iloc[:4]) == [2.0, 1.0, 1.0, 0.0]
    assert np.isnan(labels["label"].iloc[-1])  # última barra não tem futuro


def test_session_gap_gets_no_label(ohlcv):
    labels = make_labels(ohlcv, CFG_FIXED)
    step = ohlcv.index.to_series().diff().shift(-1)
    gap_bars = step[step > pd.Timedelta(minutes=2)].index
    assert len(gap_bars) > 0, "fixture deveria conter gaps de sessão"
    assert labels.loc[gap_bars, "label"].isna().all()


def test_quantile_mode_balances_classes(ohlcv):
    labels = make_labels(ohlcv, CFG_QUANTILE)["label"].dropna()
    frac_lateral = (labels == 1).mean()
    # quantil 0.33 dos |retornos| ~ 1/3 das barras LATERAL (tolerância p/ ruído)
    assert 0.23 <= frac_lateral <= 0.43, f"fração LATERAL fora do esperado: {frac_lateral:.2f}"
    # e as duas direções aparecem de forma relevante
    assert (labels == 0).mean() > 0.15
    assert (labels == 2).mean() > 0.15


def test_label_uses_only_next_bar_return():
    """Mudar barras após t+1 não pode alterar o rótulo de t."""
    df = _tiny_frame()
    labels_before = make_labels(df, CFG_FIXED)["label"]
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] = 200.0  # muda só a última barra
    labels_after = make_labels(df2, CFG_FIXED)["label"]
    pd.testing.assert_series_equal(labels_before.iloc[:3], labels_after.iloc[:3])
