"""Treina um modelo por ativo a partir do histórico gravado.

Uso avançado — no dia a dia o aplicativo treina sozinho em segundo
plano ao ver um ativo novo. Este script serve para retreinar em lote,
comparar horizontes e inspecionar métricas:

    python scripts/train.py                # todos os ativos com CSV disponível
    python scripts/train.py --symbol NVDA  # um ativo específico

Lê data/history/*.csv (gerado por download_history.py) e/ou as barras
gravadas ao vivo em data/bars/*.csv. Só salva o modelo se ele bater o
baseline de frequências em log-loss no walk-forward — modelo que não
bate baseline é ruído e não entra em produção. O modelo é salvo por
(ativo, horizonte): mude prediction.horizon_bars no config.yaml para
treinar outras janelas de previsão.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.trainer import model_path, train_symbol  # noqa: E402


def load_history(symbol: str, data_dir: Path) -> pd.DataFrame:
    frames = []
    for sub in ("history", "bars"):
        path = data_dir / sub / f"{symbol}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, index_col="time", parse_dates=["time"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="treina só este ativo")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["storage"]["data_dir"])
    horizon = cfg["prediction"]["horizon_bars"]
    minutes = horizon * cfg["bars"]["timeframe_minutes"]
    symbols = [args.symbol] if args.symbol else list(cfg["symbols"])
    print(f"Horizonte de previsão: {horizon} barra(s) = próximos {minutes} min\n")

    for sym in symbols:
        df = load_history(sym, data_dir)
        if df.empty:
            print(f"{sym}: sem histórico em {data_dir}/history|bars — pulando")
            continue

        model, metrics, msg = train_symbol(df, cfg)
        print(f"{sym}: {len(df)} barras")
        if model is not None:
            path = model_path(cfg["model"]["dir"], sym, horizon)
            model.save(path)
            print(f"  OK bateu o baseline — {msg}\n  salvo em {path}")
        else:
            print(f"  X {msg}")


if __name__ == "__main__":
    main()
