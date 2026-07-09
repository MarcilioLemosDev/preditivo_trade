"""Treinamento de modelos — usado pelo script de treino e pelo
treinamento automático em segundo plano do aplicativo.

O trader não precisa saber que isso existe: ao abrir um gráfico novo,
o app baixa o histórico pela API, treina aqui em uma thread e, se o
modelo bater o baseline, passa a usá-lo na hora ("aprendendo..." ->
"pronto" na janelinha).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from .features import build_features
from .labeling import make_labels
from .model import DirectionModel


def model_path(model_dir: str | Path, symbol: str, horizon_bars: int) -> Path:
    """Um modelo por (ativo, horizonte): prever 2 min e 6 min são tarefas diferentes."""
    return Path(model_dir) / f"{symbol}_h{horizon_bars}.joblib"


def train_symbol(df: pd.DataFrame, cfg: dict) -> tuple[DirectionModel | None, dict | None, str]:
    """Treina um ativo. Retorna (modelo_ou_None, métricas_ou_None, mensagem).

    Modelo None com métricas = treinou mas não bateu o baseline (descartado).
    Modelo None sem métricas = dados insuficientes.
    """
    horizon = int(cfg["prediction"]["horizon_bars"])
    feats = build_features(df, cfg["features"]["micro_windows"], cfg["features"]["macro_windows"])
    labels = make_labels(df, cfg["labeling"], cfg["bars"]["timeframe_minutes"], horizon)
    data = feats.join(labels["label"]).dropna()
    if len(data) < int(cfg["model"]["min_train_bars"]):
        return None, None, f"histórico insuficiente ({len(data)} amostras úteis)"

    model = DirectionModel()
    metrics = model.train(data.drop(columns="label"), data["label"],
                          calib_fraction=float(cfg["model"]["calib_fraction"]))
    if not metrics["beats_prior"]:
        return None, metrics, (
            f"não bateu o baseline (log-loss {metrics['logloss_model']:.4f} "
            f"vs {metrics['logloss_prior']:.4f}) — descartado"
        )
    return model, metrics, (
        f"log-loss {metrics['logloss_model']:.4f} vs baseline {metrics['logloss_prior']:.4f}, "
        f"acurácia balanceada {metrics['bal_acc_model']:.1%}"
    )


def train_async(symbol: str, df: pd.DataFrame, cfg: dict, on_done) -> None:
    """Treina em thread separada; on_done(symbol, model, metrics, msg) ao final."""

    def _run():
        try:
            model, metrics, msg = train_symbol(df, cfg)
        except Exception as exc:  # noqa: BLE001 - não pode derrubar o app
            model, metrics, msg = None, None, f"erro no treino: {exc}"
        on_done(symbol, model, metrics, msg)

    threading.Thread(target=_run, daemon=True, name=f"train-{symbol}").start()
