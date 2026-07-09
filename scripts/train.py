"""Treina um modelo por ativo a partir do histórico gravado.

    python scripts/train.py                # treina todos os ativos com CSV disponível
    python scripts/train.py --symbol NVDA  # treina um ativo específico

Lê data/history/*.csv (gerado por download_history.py) e/ou as barras
gravadas ao vivo em data/bars/*.csv, monta features e rótulos, valida
em walk-forward e só salva o modelo se ele bater o baseline de
frequências em log-loss. Modelo que não bate baseline é ruído — não
entra em produção.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.features import build_features  # noqa: E402
from src.labeling import make_labels  # noqa: E402
from src.model import DirectionModel  # noqa: E402


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
    model_dir = Path(cfg["model"]["dir"])
    symbols = [args.symbol] if args.symbol else list(cfg["symbols"])

    for sym in symbols:
        df = load_history(sym, data_dir)
        if len(df) < cfg["model"]["min_train_bars"]:
            print(f"{sym}: apenas {len(df)} barras (< {cfg['model']['min_train_bars']}) — pulando")
            continue

        feats = build_features(df, cfg["features"]["micro_windows"], cfg["features"]["macro_windows"])
        labels = make_labels(df, cfg["labeling"], cfg["bars"]["timeframe_minutes"])
        data = feats.join(labels["label"]).dropna()
        X, y = data.drop(columns="label"), data["label"]

        dist = y.value_counts(normalize=True).sort_index()
        print(f"\n{sym}: {len(X)} amostras | distribuição BAIXA/LAT/ALTA: "
              + " / ".join(f"{v:.0%}" for v in dist))

        model = DirectionModel()
        metrics = model.train(X, y, calib_fraction=cfg["model"]["calib_fraction"])
        print(f"  walk-forward log-loss: modelo {metrics['logloss_model']:.4f} "
              f"vs prior {metrics['logloss_prior']:.4f} | "
              f"acurácia balanceada {metrics['bal_acc_model']:.1%}")

        if metrics["beats_prior"]:
            path = model_dir / f"{sym}.joblib"
            model.save(path)
            print(f"  ✔ bateu o baseline — salvo em {path}")
        else:
            print("  ✘ NÃO bateu o baseline de frequências — modelo descartado. "
                  "Mais dados ou melhores features antes de ir para produção.")


if __name__ == "__main__":
    main()
