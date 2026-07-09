"""Ponta a ponta: replay -> pipeline -> sinais gravados, sem exceções."""

import pandas as pd

from src.datasource.replay import ReplaySource
from src.pipeline import SymbolPipeline
from src.recorder import Recorder
from tests.conftest import make_ohlcv


def test_replay_delivers_all_bars(ohlcv):
    src = ReplaySource({"TEST": ohlcv}, warmup_bars=200)
    src.connect()
    assert len(src.history("TEST", 150)) == 150
    seen = 0
    while (bar := src.poll_closed_bar("TEST")) is not None:
        seen += 1
        assert bar.symbol == "TEST"
    assert seen == len(ohlcv) - 200
    assert src.exhausted("TEST")


def test_pipeline_end_to_end(cfg, tmp_path):
    df = make_ohlcv(n_bars=800)
    warmup = 600
    src = ReplaySource({"TEST": df}, warmup_bars=warmup)
    recorder = Recorder(tmp_path)
    pipe = SymbolPipeline("TEST", cfg, src.history("TEST", warmup), recorder)

    rows = []
    while (bar := src.poll_closed_bar("TEST")) is not None:
        rows.append(pipe.on_bar(bar))

    assert len(rows) == 800 - warmup
    for row in rows:
        assert abs(sum(row["probs"].values()) - 1.0) < 1e-6
        assert row["action"] in {"COMPRA", "VENDA", "SAIR", "MANTER", "FORA"}

    bars_csv = pd.read_csv(tmp_path / "bars" / "TEST.csv")
    signals_csv = pd.read_csv(tmp_path / "signals" / "TEST.csv")
    assert len(bars_csv) == len(rows)
    assert len(signals_csv) == len(rows)
