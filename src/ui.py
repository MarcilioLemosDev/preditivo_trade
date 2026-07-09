"""Interfaces de exibição dos sinais.

OverlayUI: janelinha compacta, escura e sempre-no-topo (tkinter, nativo
do Python — nada a instalar), pensada para ocupar poucos pixels num
canto do monitor sem atrapalhar a leitura do gráfico. Uma linha por
ativo: preço, probabilidades, sugestão e o modo do modelo. As linhas
entram e saem sozinhas conforme o trader abre/fecha gráficos no MT5.
Quem age é o humano; a janela só informa.

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

# modo do modelo por ativo: básico (frequências), aprendendo (treino em
# segundo plano) ou pronto (modelo treinado e aprovado)
MODE_COLORS = {"básico": "#9ca3af", "aprendendo": "#f59e0b", "pronto": "#22c55e"}

_COLUMNS = ("symbol", "close", "alta", "lat", "baixa", "action", "mode")
_HEADERS = ("ATIVO", "PREÇO", "ALTA", "LAT", "BAIXA", "SUGESTÃO", "MODO")


class ConsoleUI:
    def __init__(self, subtitle: str = ""):
        self._subtitle = subtitle
        self._message = None

    def start(self, symbols: list[str] | None = None) -> None:
        if self._subtitle:
            print(self._subtitle)
        if symbols:
            print(f"Acompanhando: {', '.join(symbols)}")

    def ensure_row(self, symbol: str) -> None:
        pass

    def remove_row(self, symbol: str) -> None:
        print(f"{symbol}: gráfico fechado, deixei de acompanhar")

    def set_message(self, text: str) -> None:
        if text and text != self._message:
            print(text)
        self._message = text

    def update(self, symbol: str, row: dict) -> None:
        print(
            f"[{row['time']:%H:%M}] {symbol:<8} {row['close']:>12.4f} | "
            f"ALTA {row['probs']['ALTA']:>4.0%}  LAT {row['probs']['LATERAL']:>4.0%}  "
            f"BAIXA {row['probs']['BAIXA']:>4.0%} | {row['action']:<6} "
            f"[{row.get('mode', 'básico')}] {row['reason']}"
        )

    def run(self, tick, interval_ms: int) -> None:
        import time
        while True:
            if tick() is False:
                break
            time.sleep(interval_ms / 1000)


class OverlayUI:
    def __init__(self, always_on_top: bool = True, font_size: int = 9, subtitle: str = ""):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title(f"preditivo_trade — {subtitle}" if subtitle else "preditivo_trade")
        self.root.configure(bg="#111827")
        self.root.attributes("-topmost", bool(always_on_top))
        self.root.resizable(False, False)
        self._font = ("Consolas", int(font_size))
        self._rows: dict[str, dict] = {}

        self._msg = tk.Label(self.root, text="", font=self._font, fg="#fbbf24",
                             bg="#111827", padx=6, pady=4, anchor="w")

    def start(self, symbols: list[str] | None = None) -> None:
        tk = self._tk
        for col, text in enumerate(_HEADERS):
            tk.Label(self.root, text=text, font=(self._font[0], self._font[1], "bold"),
                     fg="#6b7280", bg="#111827", padx=6).grid(row=0, column=col, sticky="w")
        for sym in symbols or []:
            self.ensure_row(sym)

    def ensure_row(self, symbol: str) -> None:
        if symbol in self._rows:
            return
        tk = self._tk
        labels = {}
        row_index = len(self._rows) + 1
        for col, key in enumerate(_COLUMNS):
            lbl = tk.Label(self.root, text="—", font=self._font, fg="#e5e7eb",
                           bg="#111827", padx=6, anchor="w")
            lbl.grid(row=row_index, column=col, sticky="w")
            labels[key] = lbl
        labels["symbol"].config(text=symbol, fg="#93c5fd")
        self._rows[symbol] = labels

    def remove_row(self, symbol: str) -> None:
        labels = self._rows.pop(symbol, None)
        if labels is None:
            return
        for lbl in labels.values():
            lbl.destroy()
        # reposiciona as linhas restantes
        for i, row_labels in enumerate(self._rows.values(), start=1):
            for col, key in enumerate(_COLUMNS):
                row_labels[key].grid(row=i, column=col, sticky="w")

    def set_message(self, text: str) -> None:
        if text:
            self._msg.config(text=text)
            self._msg.grid(row=200, column=0, columnspan=len(_COLUMNS), sticky="w")
        else:
            self._msg.grid_remove()

    def update(self, symbol: str, row: dict) -> None:
        self.ensure_row(symbol)
        labels = self._rows[symbol]
        probs = row["probs"]
        mode = row.get("mode", "básico")
        labels["close"].config(text=f"{row['close']:.4f}")
        labels["alta"].config(text=f"{probs['ALTA']:.0%}", fg="#22c55e")
        labels["lat"].config(text=f"{probs['LATERAL']:.0%}", fg="#9ca3af")
        labels["baixa"].config(text=f"{probs['BAIXA']:.0%}", fg="#ef4444")
        labels["action"].config(text=row["action"], fg=ACTION_COLORS.get(row["action"], "#e5e7eb"))
        labels["mode"].config(text=mode, fg=MODE_COLORS.get(mode, "#9ca3af"))

    def run(self, tick, interval_ms: int) -> None:
        def _loop():
            if tick() is False:
                self.root.destroy()
                return
            self.root.after(interval_ms, _loop)

        self.root.after(0, _loop)
        self.root.mainloop()
