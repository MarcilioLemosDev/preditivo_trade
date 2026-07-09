"""Carregamento e validação da configuração central (config.yaml)."""

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_REQUIRED_SECTIONS = ("prediction", "symbols", "bars", "labeling", "features", "signals",
                      "model", "ui", "storage")


def load_config(path: str | Path = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml sem as seções obrigatórias: {missing}")
    if not cfg["symbols"]:
        raise ValueError("config.yaml precisa de ao menos um símbolo em 'symbols'")
    return cfg
