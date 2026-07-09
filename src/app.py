"""Aplicativo principal: acompanha em tempo real os gráficos abertos no MT5.

Para o trader: duplo clique em INICIAR.bat. O programa descobre sozinho
quais gráficos estão abertos no MetaTrader 5, baixa o histórico pela
API, treina o modelo em segundo plano na primeira vez que vê um ativo
("aprendendo..." na janelinha) e passa a exibir probabilidades e
sugestões a cada barra de 2 minutos. Abrir/fechar gráficos no MT5
adiciona/remove ativos da janelinha automaticamente.

Uso técnico direto:

    python -m src.app --source mt5                                  # ao vivo
    python -m src.app --source replay --replay-dir data/history --console --fast
"""

from __future__ import annotations

import argparse
import queue
from pathlib import Path

from .config import load_config
from .datasource.replay import ReplaySource
from .model import DirectionModel
from .pipeline import SymbolPipeline
from .recorder import Recorder
from .trainer import model_path, train_async
from .ui import ConsoleUI, OverlayUI


def _build_ui(cfg, args, subtitle: str):
    if args.console or cfg["ui"]["mode"] == "console":
        return ConsoleUI(subtitle=subtitle)
    return OverlayUI(cfg["ui"]["always_on_top"], cfg["ui"]["font_size"], subtitle=subtitle)


def _load_model(cfg, symbol: str) -> DirectionModel | None:
    path = model_path(cfg["model"]["dir"], symbol, cfg["prediction"]["horizon_bars"])
    return DirectionModel.load(path) if path.exists() else None


# ----------------------------------------------------------------------
def run_live(cfg, args, ui) -> None:
    from .chart_watch import detect_open_charts
    from .datasource.mt5_source import MT5Source

    source = MT5Source(cfg)
    source.connect()
    recorder = Recorder(cfg["storage"]["data_dir"])
    horizon = cfg["prediction"]["horizon_bars"]

    pipelines: dict[str, SymbolPipeline] = {}
    training: set[str] = set()
    failed: set[str] = set()
    results: queue.Queue = queue.Queue()

    def on_trained(symbol, model, metrics, msg):
        results.put((symbol, model, metrics, msg))

    def refresh_charts():
        detected = detect_open_charts()
        for sym in detected:
            if sym in pipelines or sym in failed:
                continue
            try:
                source.ensure_symbol(sym)
                history = source.history(sym, cfg["bars"]["history_bars"])
            except ValueError as exc:
                print(f"{sym}: ignorando gráfico ({exc})")
                failed.add(sym)
                continue
            model = _load_model(cfg, sym)
            pipelines[sym] = SymbolPipeline(sym, cfg, history, recorder, model)
            ui.ensure_row(sym)
            print(f"{sym}: acompanhando ({len(history)} barras de histórico, "
                  f"{'modelo pronto' if model else 'modo básico'})")
            if model is None and cfg["model"].get("auto_train", True) and sym not in training:
                train_hist = source.history(sym, cfg["model"]["train_bars"])
                if len(train_hist) >= cfg["model"]["min_train_bars"]:
                    training.add(sym)
                    print(f"{sym}: aprendendo em segundo plano "
                          f"({len(train_hist)} barras)...")
                    train_async(sym, train_hist, cfg, on_trained)
        for sym in list(pipelines):
            if sym not in detected:
                del pipelines[sym]
                ui.remove_row(sym)
        ui.set_message("" if pipelines
                       else "Abra um gráfico no MetaTrader 5 — eu acompanho sozinho.")

    def drain_training_results():
        while True:
            try:
                sym, model, metrics, msg = results.get_nowait()
            except queue.Empty:
                return
            training.discard(sym)
            if model is not None:
                model.save(model_path(cfg["model"]["dir"], sym, horizon))
                if sym in pipelines:
                    pipelines[sym].model = model
                print(f"{sym}: modelo pronto — {msg}")
            else:
                print(f"{sym}: seguindo no modo básico — {msg}")

    poll_s = float(cfg["bars"]["poll_seconds"])
    scan_ticks = max(1, round(cfg["bars"].get("chart_scan_seconds", 5) / poll_s))
    ticks = 0

    def tick():
        nonlocal ticks
        if ticks % scan_ticks == 0:
            refresh_charts()
        drain_training_results()
        for sym, pipe in list(pipelines.items()):
            bar = source.poll_closed_bar(sym)
            if bar is not None:
                row = pipe.on_bar(bar)
                if sym in training:
                    row["mode"] = "aprendendo"
                ui.update(sym, row)
        ticks += 1
        return True

    ui.start()
    try:
        ui.run(tick, int(poll_s * 1000))
    finally:
        source.close()


# ----------------------------------------------------------------------
def run_replay(cfg, args, ui) -> None:
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
    source = ReplaySource.from_csv(paths, warmup_bars=cfg["bars"]["history_bars"])
    source.connect()
    recorder = Recorder(cfg["storage"]["data_dir"])
    symbols = list(paths)

    pipelines = {}
    for sym in symbols:
        model = _load_model(cfg, sym)
        history = source.history(sym, cfg["bars"]["history_bars"])
        pipelines[sym] = SymbolPipeline(sym, cfg, history, recorder, model)
        print(f"{sym}: {len(history)} barras de histórico, "
              f"{'modelo pronto' if model else 'modo básico'}")

    def tick():
        got_any = False
        for sym, pipe in pipelines.items():
            bar = source.poll_closed_bar(sym)
            if bar is not None:
                got_any = True
                ui.update(sym, pipe.on_bar(bar))
        if not got_any and all(source.exhausted(s) for s in symbols):
            print("Replay concluído.")
            return False
        return True

    interval_ms = 1 if args.fast else int(cfg["bars"]["poll_seconds"] * 1000)
    ui.start(symbols)
    try:
        ui.run(tick, interval_ms)
    finally:
        source.close()


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="preditivo_trade — acompanhamento em tempo real")
    parser.add_argument("--source", choices=["mt5", "replay"], default="mt5")
    parser.add_argument("--replay-dir", default="data/history")
    parser.add_argument("--config", default=None)
    parser.add_argument("--console", action="store_true", help="usa terminal em vez da janelinha")
    parser.add_argument("--fast", action="store_true", help="replay em velocidade máxima")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    minutes = cfg["prediction"]["horizon_bars"] * cfg["bars"]["timeframe_minutes"]
    ui = _build_ui(cfg, args, subtitle=f"previsão: próximos {minutes} min")

    if args.source == "mt5":
        run_live(cfg, args, ui)
    else:
        run_replay(cfg, args, ui)


if __name__ == "__main__":
    main()
