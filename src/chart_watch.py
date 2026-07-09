"""Detecção automática dos gráficos abertos no MT5.

Para o trader, o programa "vê a tela": ele descobre sozinho quais
gráficos estão abertos no MetaTrader 5 e passa a acompanhá-los, sem
nenhuma configuração. Tecnicamente, lê os títulos das janelas do
terminal via API do Windows (as janelas de gráfico do MT5 chamam-se
"SIMBOLO,TEMPO", ex.: "XAUUSD,M2"; o gráfico ativo maximizado aparece
no título principal como "[SIMBOLO,TEMPO]").

O timeframe que o trader usa no gráfico dele é irrelevante: serve só
para identificar o ativo — o modelo analisa sempre barras de 2 minutos
vindas da API, independente do zoom ou período exibido na tela.
"""

from __future__ import annotations

import re
import sys

_TIMEFRAME = r"(?:M\d+|H\d+|D\d*|W\d*|MN\d*|Daily|Weekly|Monthly)"
_SYMBOL = r"([\w.#&@!$\-]+)"
_CHILD_RE = re.compile(rf"^{_SYMBOL},{_TIMEFRAME}(?:\s|$)")
_ACTIVE_RE = re.compile(rf"\[{_SYMBOL},{_TIMEFRAME}\]")


def parse_chart_title(title: str) -> str | None:
    """'XAUUSD,M2' -> 'XAUUSD'; título que não é de gráfico -> None."""
    m = _CHILD_RE.match(title.strip())
    return m.group(1) if m else None


def parse_active_chart(title: str) -> str | None:
    """Extrai '[NVDA,M2]' do título principal do terminal -> 'NVDA'."""
    m = _ACTIVE_RE.search(title)
    return m.group(1) if m else None


def detect_open_charts() -> list[str]:
    """Símbolos dos gráficos abertos no MT5, gráfico ativo primeiro.

    Retorna lista vazia fora do Windows ou sem terminal/gráfico aberto.
    """
    if sys.platform != "win32":
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def window_text(hwnd) -> str:
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def window_class(hwnd) -> str:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    active: list[str] = []
    found: list[str] = []

    def on_child(hwnd, _):
        sym = parse_chart_title(window_text(hwnd))
        if sym:
            found.append(sym)
        return True

    on_child_cb = CB(on_child)

    def on_top(hwnd, _):
        title = window_text(hwnd)
        # identifica o terminal pela classe da janela (estável mesmo em
        # terminais rebatizados por corretoras) ou pelo título padrão
        if "MetaTrader 5" in title or window_class(hwnd).startswith("MetaQuotes::MetaTrader"):
            sym = parse_active_chart(title)
            if sym:
                active.append(sym)
            user32.EnumChildWindows(hwnd, on_child_cb, 0)
        return True

    user32.EnumWindows(CB(on_top), 0)

    seen: set[str] = set()
    ordered = []
    for sym in active + found:
        if sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered
