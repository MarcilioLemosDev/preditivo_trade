import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_ohlcv(n_bars: int = 1200, seed: int = 7, start: str = "2026-06-01 13:30",
               with_session_gaps: bool = True) -> pd.DataFrame:
    """Random walk em barras de 2 min, com gaps de sessão como no mundo real."""
    rng = np.random.default_rng(seed)
    # sessões de ~6,5h (195 barras de 2 min), com pulo para o dia seguinte
    times = []
    t = pd.Timestamp(start, tz="UTC")
    per_session = 195 if with_session_gaps else n_bars
    while len(times) < n_bars:
        session_start = t
        for i in range(min(per_session, n_bars - len(times))):
            times.append(session_start + pd.Timedelta(minutes=2 * i))
        t = session_start + pd.Timedelta(days=1)
    index = pd.DatetimeIndex(times[:n_bars], name="time")

    ret = rng.normal(0, 0.001, n_bars)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0008, n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100, 5000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def cfg() -> dict:
    from src.config import load_config
    return load_config()
