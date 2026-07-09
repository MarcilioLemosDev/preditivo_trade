"""Interfaces de exibição dos sinais.

OverlayUI: janelinha compacta, escura e sempre-no-topo (tkinter, nativo
do Python — nada a instalar), pensada para ocupar poucos pixels num
canto do monitor sem atrapalhar a leitura do gráfico. Uma linha por
ativo: preço, probabilidades e a sugestão do momento. Quem age é o
humano; a janela só informa.

ConsoleUI: mesma informação em texto puro — útil em desenvolvimento,
replay e servidores sem interface gráfica.
"""

from __future__ import annotations

ACTION_COLORS = {
    "COMPRA": "#22c55e",
    "VENDA": "#ef4444",
    "SAIR": "#f59e0b",
    "MANTER": "#38bdf8",
    "FORA": "#9ca3af",
}


class ConsoleUI:
    def start(self, symbols: list[str]) -> None:
        print(f"Acompanhando: {', '.join(symbols)}")

    def update(self, symbol: str, row: dict) -> None:
        print(
            f"[{row['time']:%H:%M}] {symbol:<6} {row['close']:>12.4f} | "
            f"ALTA {row['probs']['ALTA']:>4.0%}  LAT {row['probs']['LATERAL']:>4.0%}  "
            f"BAIXA {row['probs']['BAIXA']:>4.0%} | {row['action']:<6} {row['reason']}"
        )

    def run(self, tick, interval_ms: int) -> None:
        import time
        while True:
            if tick() is False:
                break
            time.sleep(interval_ms / 1000)


class OverlayUI:
    def __init__(self, always_on_top: bool = True, font_size: int = 9):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("preditivo_trade")
        self.root.configure(bg="#111827")
        self.root.attributes("-topmost", bool(always_on_top))
        self.root.resizable(False, False)
        self._font = ("Consolas", int(font_size))
        self._rows: dict[str, dict] = {}

    def start(self, symbols: list[str]) -> None:
        tk = self._tk
        header = ("ATIVO", "PREÇO", "ALTA", "LAT", "BAIXA", "SUGESTÃO")
        for col, text in enumerate(header):
            tk.Label(self.root, text=text, font=(*self._font[:1], self._font[1], "bold"),
                     fg="#6b7280", bg="#111827", padx=6).grid(row=0, column=col, sticky="w")
        for i, sym in enumerate(symbols, start=1):
            labels = {}
            for col, key in enumerate(("symbol", "close", "alta", "lat", "baixa", "action")):
                lbl = tk.Label(self.root, text="—", font=self._font, fg="#e5e7eb",
                               bg="#111827", padx=6, anchor="w")
                lbl.grid(row=i, column=col, sticky="w")
                labels[key] = lbl
            labels["symbol"].config(text=sym, fg="#93c5fd")
            self._rows[sym] = labels

    def update(self, symbol: str, row: dict) -> None:
        labels = self._rows[symbol]
        probs = row["probs"]
        labels["close"].config(text=f"{row['close']:.4f}")
        labels["alta"].config(text=f"{probs['ALTA']:.0%}", fg="#22c55e")
        labels["lat"].config(text=f"{probs['LATERAL']:.0%}", fg="#9ca3af")
        labels["baixa"].config(text=f"{probs['BAIXA']:.0%}", fg="#ef4444")
        labels["action"].config(text=row["action"], fg=ACTION_COLORS.get(row["action"], "#e5e7eb"))

    def run(self, tick, interval_ms: int) -> None:
        def _loop():
            if tick() is False:
                self.root.destroy()
                return
            self.root.after(interval_ms, _loop)

        self.root.after(0, _loop)
        self.root.mainloop()
