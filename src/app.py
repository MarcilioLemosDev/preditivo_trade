"""Aplicativo principal: acompanha os ativos em tempo real.

Uso na máquina do trader (Windows, terminal MT5 aberto e logado):

    python -m src.app --source mt5

Desenvolvimento/simulação com histórico gravado (qualquer sistema):

    python -m src.app --source replay --replay-dir data/history --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .datasource.replay import ReplaySource
from .model import DirectionModel
from .pipeline import SymbolPipeline
from .recorder import Recorder
from .ui import ConsoleUI, OverlayUI


def build_source(args, cfg):
    if args.source == "mt5":
        from .datasource.mt5_source import MT5Source  # import tardio: só existe no Windows
        return MT5Source(cfg)
    paths = {}
    for sym in cfg["symbols"]:
        path = Path(args.replay_dir) / f"{sym}.csv"
        if path.exists():
            paths[sym] = str(path)
    if not paths:
        raise SystemExit(
            f"Nenhum CSV encontrado em {args.replay_dir}. "
            "Rode scripts/download_history.py na máquina com MT5 primeiro."
        )
    return ReplaySource.from_csv(paths, warmup_bars=cfg["bars"]["history_bars"])


def main() -> None:
    parser = argparse.ArgumentParser(description="preditivo_trade — acompanhamento em tempo real")
    parser.add_argument("--source", choices=["mt5", "replay"], default="mt5")
    parser.add_argument("--replay-dir", default="data/history")
    parser.add_argument("--config", default=None)
    parser.add_argument("--console", action="store_true", help="usa terminal em vez da janelinha")
    parser.add_argument("--fast", action="store_true", help="replay em velocidade máxima")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    source = build_source(args, cfg)
    source.connect()

    recorder = Recorder(cfg["storage"]["data_dir"])
    model_dir = Path(cfg["model"]["dir"])
    symbols = (list(source._frames) if isinstance(source, ReplaySource)
               else list(cfg["symbols"]))

    pipelines: dict[str, SymbolPipeline] = {}
    for sym in symbols:
        model_path = model_dir / f"{sym}.joblib"
        model = DirectionModel.load(model_path) if model_path.exists() else None
        history = source.history(sym, cfg["bars"]["history_bars"])
        pipelines[sym] = SymbolPipeline(sym, cfg, history, recorder, model)
        status = "modelo treinado" if model else "baseline (sem modelo treinado ainda)"
        print(f"{sym}: {len(history)} barras de histórico, {status}")

    if args.console or cfg["ui"]["mode"] == "console":
        ui = ConsoleUI()
    else:
        ui = OverlayUI(cfg["ui"]["always_on_top"], cfg["ui"]["font_size"])
    ui.start(symbols)

    def tick():
        got_any = False
        for sym, pipe in pipelines.items():
            bar = source.poll_closed_bar(sym)
            if bar is not None:
                got_any = True
                ui.update(sym, pipe.on_bar(bar))
        if isinstance(source, ReplaySource) and not got_any:
            if all(source.exhausted(s) for s in symbols):
                print("Replay concluído.")
                return False
        return True

    interval_ms = 1 if (args.fast and args.source == "replay") \
        else int(cfg["bars"]["poll_seconds"] * 1000)
    try:
        ui.run(tick, interval_ms)
    finally:
        source.close()


if __name__ == "__main__":
    main()
