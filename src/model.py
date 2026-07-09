"""Modelos preditivos: baseline de frequências e LightGBM calibrado.

Dois princípios inegociáveis:
1. Validação sempre walk-forward (treina no passado, testa no futuro).
2. O LightGBM só é considerado útil se bater o baseline de frequências
   em log-loss no walk-forward — senão estamos vendendo ruído.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from .labeling import CLASS_NAMES

CLASSES = np.array([0, 1, 2])  # BAIXA, LATERAL, ALTA


def _proba_dict(p: np.ndarray) -> dict[str, float]:
    return {CLASS_NAMES[c]: float(p[i]) for i, c in enumerate(CLASSES)}


class RollingPriorsModel:
    """Baseline: frequência das classes nas últimas N barras rotuladas.

    É o modelo em produção enquanto não existe LightGBM treinado para o
    ativo — honesto por construção: sem histórico, devolve incerteza
    uniforme e nunca dispara sinal.
    """

    def __init__(self, window: int = 500):
        self._labels: deque[int] = deque(maxlen=window)

    def update(self, label: int) -> None:
        self._labels.append(int(label))

    def predict_proba_dict(self, _features_row: pd.DataFrame | None = None) -> dict[str, float]:
        n = len(self._labels)
        if n < 30:  # amostra pequena demais: incerteza total
            return _proba_dict(np.full(3, 1 / 3))
        counts = np.bincount(np.asarray(self._labels), minlength=3).astype(float)
        return _proba_dict((counts + 1) / (n + 3))  # suavização de Laplace


class DirectionModel:
    """LightGBM multiclasse com calibração isotônica por classe."""

    def __init__(self, params: dict | None = None):
        self.params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            **(params or {}),
        }
        self.booster = None
        self.calibrators: list[IsotonicRegression] | None = None
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series, calib_fraction: float = 0.2,
              n_splits: int = 5) -> dict:
        """Walk-forward + ajuste final + calibração. Retorna métricas."""
        import lightgbm as lgb

        X = X.astype(float)
        y = y.astype(int)
        self.feature_names = list(X.columns)

        # ---- avaliação walk-forward (modelo vs baseline de priors) ----
        rows = []
        for tr_idx, te_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            clf = lgb.LGBMClassifier(**self.params)
            clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = clf.predict_proba(X.iloc[te_idx])
            prior = np.bincount(y.iloc[tr_idx], minlength=3) / len(tr_idx)
            prior_mat = np.tile(prior, (len(te_idx), 1))
            rows.append({
                "logloss_model": log_loss(y.iloc[te_idx], proba, labels=CLASSES),
                "logloss_prior": log_loss(y.iloc[te_idx], prior_mat, labels=CLASSES),
                "bal_acc_model": balanced_accuracy_score(y.iloc[te_idx], proba.argmax(1)),
            })
        wf = pd.DataFrame(rows).mean().to_dict()
        wf["beats_prior"] = bool(wf["logloss_model"] < wf["logloss_prior"])
        wf["n_samples"] = len(X)

        # ---- ajuste final: treino na fatia inicial, calibração no fim ----
        cut = int(len(X) * (1 - calib_fraction))
        clf = lgb.LGBMClassifier(**self.params)
        clf.fit(X.iloc[:cut], y.iloc[:cut])
        raw = clf.predict_proba(X.iloc[cut:])
        y_cal = y.iloc[cut:].to_numpy()
        self.calibrators = []
        for i, c in enumerate(CLASSES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(raw[:, i], (y_cal == c).astype(float))
            self.calibrators.append(iso)
        self.booster = clf
        return wf

    # ------------------------------------------------------------------
    def _calibrated(self, raw: np.ndarray) -> np.ndarray:
        cal = np.column_stack([iso.predict(raw[:, i]) for i, iso in enumerate(self.calibrators)])
        return cal / cal.sum(axis=1, keepdims=True)

    def predict_proba_dict(self, features_row: pd.DataFrame) -> dict[str, float]:
        if self.booster is None:
            raise RuntimeError("modelo não treinado")
        x = features_row[self.feature_names].astype(float)
        return _proba_dict(self._calibrated(self.booster.predict_proba(x))[-1])

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "DirectionModel":
        return joblib.load(path)
