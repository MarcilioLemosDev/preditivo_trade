"""Baixa histórico de barras M2 do MT5 para todos os ativos do config.

Rodar na máquina Windows do trader, com o terminal MT5 aberto e logado:

    python scripts/download_history.py --bars 20000

Nota: o MT5 entrega o histórico completo independente do zoom do
gráfico — não há limitação de "print da tela". O limite real é a
profundidade de histórico que a corretora disponibiliza por símbolo
(configurável em Ferramentas > Opções > Gráficos > Máx. de barras).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.datasource.mt5_source import MT5Source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=20000, help="barras M2 por ativo (~28 pregões de ação)")
    parser.add_argument("--out", default="data/history")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = MT5Source(cfg)
    source.connect()
    try:
        for sym in cfg["symbols"]:
            df = source.history(sym, args.bars)
            path = out_dir / f"{sym}.csv"
            df.to_csv(path)
            if df.empty:
                print(f"{sym}: NENHUMA barra retornada — confira o nome do símbolo no config.yaml")
            else:
                print(f"{sym}: {len(df)} barras ({df.index[0]} a {df.index[-1]}) -> {path}")
    finally:
        source.close()


if __name__ == "__main__":
    main()
