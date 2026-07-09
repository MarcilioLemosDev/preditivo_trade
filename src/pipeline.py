"""Pipeline por ativo: barra fechada -> features -> probabilidades -> sinal.

Cada ativo tem seu próprio estado (histórico, modelo, posição sugerida).
Enquanto não existe modelo LightGBM treinado para o ativo, roda o
baseline de frequências — que por construção quase nunca dispara sinal,
mas mantém o sistema inteiro funcionando e gravando dados para o treino.

A previsão sempre olha `horizon_bars` barras à frente (1 = 2 min,
2 = 4 min...), conforme configurado pelo trader.
"""

from __future__ import annotations

import pandas as pd

from .datasource.base import Bar
from .features import build_features, warmup_bars
from .labeling import make_labels
from .model import DirectionModel, RollingPriorsModel
from .recorder import Recorder
from .signals import COMPRA, SAIR, VENDA, decide


class SymbolPipeline:
    def __init__(self, symbol: str, cfg: dict, history: pd.DataFrame,
                 recorder: Recorder, model: DirectionModel | None = None):
        self.symbol = symbol
        self.cfg = cfg
        self.horizon = int(cfg["prediction"]["horizon_bars"])
        self.df = history.copy()
        self.recorder = recorder
        self.model = model
        self.baseline = RollingPriorsModel(window=cfg["labeling"]["window"])
        self.position: str | None = None  # posição SUGERIDA (quem executa é o humano)

        # tamanho de histórico mantido em memória: o bastante p/ features + theta
        self._keep = max(
            warmup_bars(cfg["features"]["macro_windows"]),
            cfg["labeling"]["window"] + self.horizon + 10,
        ) + 50

        # aquece o baseline com os rótulos do histórico carregado
        if len(self.df) > 100:
            labels = self._labels()
            for lbl in labels["label"].dropna():
                self.baseline.update(int(lbl))

    def _labels(self) -> pd.DataFrame:
        return make_labels(self.df, self.cfg["labeling"],
                           self.cfg["bars"]["timeframe_minutes"], self.horizon)

    @property
    def mode(self) -> str:
        return "pronto" if self.model is not None else "básico"

    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> dict:
        """Processa uma barra recém-fechada e devolve a linha para a UI."""
        self.recorder.record_bar(bar)
        self.df = pd.concat([self.df, bar.as_row()]).tail(self._keep)
        self.df = self.df[~self.df.index.duplicated(keep="last")]

        # com o fechamento desta barra, o rótulo da barra t-h ficou
        # conhecido -> alimenta o baseline (aprendizado contínuo honesto)
        labels = self._labels()
        idx = -(1 + self.horizon)
        prev_label = labels["label"].iloc[idx] if len(labels) >= -idx else float("nan")
        if pd.notna(prev_label):
            self.baseline.update(int(prev_label))

        # previsão para h barras à frente usando só o que se sabe até agora
        feats = build_features(
            self.df,
            self.cfg["features"]["micro_windows"],
            self.cfg["features"]["macro_windows"],
        ).tail(1)
        if self.model is not None and not feats.isna().any(axis=1).iloc[-1]:
            probs = self.model.predict_proba_dict(feats)
        else:
            probs = self.baseline.predict_proba_dict(feats)

        sig = decide(probs, self.position, self.cfg["signals"])
        if sig.action == COMPRA:
            self.position = "COMPRADO"
        elif sig.action == VENDA:
            self.position = "VENDIDO"
        elif sig.action == SAIR:
            self.position = None

        self.recorder.record_signal(self.symbol, bar.time, bar.close, probs,
                                    sig.action, sig.reason)
        return {
            "time": bar.time,
            "close": bar.close,
            "probs": probs,
            "action": sig.action,
            "reason": sig.reason,
            "mode": self.mode,
        }
