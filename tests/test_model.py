import numpy as np
import pandas as pd

from src.features import build_features
from src.labeling import make_labels
from src.model import DirectionModel, RollingPriorsModel

CFG_LBL = {"mode": "quantile", "lateral_quantile": 0.33, "window": 300, "fixed_theta_pct": 0.0005}


def test_rolling_priors_starts_uniform_and_tracks():
    m = RollingPriorsModel(window=100)
    assert m.predict_proba_dict() == {"BAIXA": 1 / 3, "LATERAL": 1 / 3, "ALTA": 1 / 3}
    for _ in range(60):
        m.update(2)  # só ALTA
    probs = m.predict_proba_dict()
    assert probs["ALTA"] > 0.9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_direction_model_train_predict_save_load(ohlcv, tmp_path):
    feats = build_features(ohlcv)
    labels = make_labels(ohlcv, CFG_LBL)
    data = feats.join(labels["label"]).dropna()
    X, y = data.drop(columns="label"), data["label"]

    model = DirectionModel(params={"n_estimators": 40})
    metrics = model.train(X, y, n_splits=3)
    assert metrics["n_samples"] == len(X)
    assert metrics["logloss_model"] > 0

    probs = model.predict_proba_dict(X.tail(1))
    assert set(probs) == {"BAIXA", "LATERAL", "ALTA"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in probs.values())

    path = tmp_path / "m.joblib"
    model.save(path)
    reloaded = DirectionModel.load(path)
    assert reloaded.predict_proba_dict(X.tail(1)) == probs


def test_random_walk_should_not_beat_prior_by_much(ohlcv):
    """Dados de passeio aleatório não têm sinal: o modelo não deve 'ganhar' com folga.

    Guarda-corpo de sanidade estatística: se um dia este teste falhar com
    vantagem grande do modelo, o mais provável é vazamento de dados.
    """
    feats = build_features(ohlcv)
    labels = make_labels(ohlcv, CFG_LBL)
    data = feats.join(labels["label"]).dropna()
    model = DirectionModel(params={"n_estimators": 40})
    metrics = model.train(data.drop(columns="label"), data["label"], n_splits=3)
    edge = metrics["logloss_prior"] - metrics["logloss_model"]
    assert edge < 0.05, f"modelo 'venceu' ruído puro por {edge:.4f} — suspeita de vazamento"
