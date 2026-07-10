"""Simulador de operações (paper trading) — SÓ dentro da interface.

Não envia nenhuma ordem: apenas simula entradas e saídas com base nas
probabilidades do modelo e contabiliza o desempenho em dólares, para
responder à pergunta que importa: *o modelo ganharia dinheiro?*

Regras de negócio definidas pelo trader:
- Até 2 entradas e 2 saídas por janela de 10 minutos (teto de atividade).
  Cada pedaço vendido numa saída parcial conta como uma "saída".
- Pode segurar a posição por até 10 minutos (5 barras de 2 min).
- Nunca entrar e sair em menos de 40 segundos (tempo mínimo de posição).
- Cada operação usa entre US$ 2.000 e US$ 4.000 (proporcional à
  confiança do modelo no sinal).
- SAÍDA PARCIAL (scaling out): o modelo não precisa zerar de uma vez.
  Conforme o risco contra a posição sobe (probabilidade contrária
  crescendo), ele alivia em pedaços; pressão forte ou fim do prazo zera
  o restante.
- Uma "operação" = um ciclo completo (entrada até zerar tudo, em um ou
  vários pedaços). Ganha ou perde pelo resultado LÍQUIDO somado.
- Meta: ao atingir 20 operações, ter winrate >= 68%.

Persistência e auditoria: cada operação recebe um `op_id` único e
crescente. As operações completas e cada pedaço vendido são gravados em
disco (ver recorder.py), permitindo conferir depois se a soma dos
pedaços bate com o P&L relatado da operação. O placar (nº de operações,
winrate, P&L) é CUMULATIVO entre sessões: ao iniciar, o pipeline semeia
o PaperTrader com o resumo lido do disco.

Honestidade estatística (inegociável): o winrate NÃO é forçado. O
simulador só entra quando a confiança do modelo passa de um limiar
(seletividade) — a única alavanca legítima para perseguir os 68%.
Tradar menos e melhor. Se nem no ajuste mais seletivo os 68% aparecem,
o veredito honesto é que o modelo ainda não é bom o bastante naquele
ativo/horizonte — e é para isso que este simulador existe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

LONG, SHORT = "LONG", "SHORT"

# defaults usados quando a seção `simulation` não está no config.yaml
DEFAULTS = {
    "enabled": True,
    "entry_prob": 0.62,          # confiança mínima p/ ENTRAR (mais seletivo que os sinais exibidos)
    "entry_margin": 0.30,        # vantagem mínima da direção vencedora sobre a oposta
    "partial_exit_prob": 0.45,   # pressão contrária MODERADA -> alivia um pedaço
    "partial_fraction": 0.5,     # fração do que resta vendida em cada saída parcial
    "exit_prob": 0.55,           # pressão contrária FORTE -> zera o restante
    "min_hold_seconds": 40,      # regra: não entrar e sair em menos de 40s
    "window_seconds": 600,       # janela de 10 min para o teto de atividade
    "max_entries_per_window": 2,
    "max_exits_per_window": 2,   # saídas (inclui parciais) por janela
    "max_hold_minutes": 10,      # tempo máximo segurando a posição (5 barras de 2 min)
    "size_min_usd": 2000.0,      # capital mínimo por operação
    "size_max_usd": 4000.0,      # capital máximo por operação
    "cost_pct": 0.0,             # custo por operação (spread+corretagem) em fração do capital
    "goal_trades": 20,
    "goal_winrate": 0.68,
}


@dataclass
class OpenPosition:
    op_id: int
    side: str
    entry_time: datetime
    entry_price: float
    initial_size: float
    remaining: float
    entry_prob: float
    bars_held: int = 0
    realized_pnl: float = 0.0    # lucro/perda já realizado nos pedaços vendidos
    clips: int = 0               # nº de saídas (parciais + final)
    last_reason: str = ""


@dataclass
class Clip:
    """Um pedaço vendido (saída parcial ou final) de uma operação."""
    op_id: int
    symbol: str
    clip_no: int
    side: str
    time: datetime
    entry_price: float
    price: float
    chunk_usd: float
    chunk_pnl: float
    remaining_usd: float
    reason: str


@dataclass
class ClosedTrade:
    op_id: int
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float              # tamanho inicial da operação
    pnl_usd: float               # resultado líquido somando todos os pedaços
    won: bool
    reason: str
    clips: int


@dataclass
class Seed:
    """Resumo lido do disco para tornar o placar cumulativo entre sessões."""
    n_trades: int = 0
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    next_op_id: int = 0


class PaperTrader:
    def __init__(self, symbol: str, cfg: dict, seed: Seed | None = None):
        self.symbol = symbol
        self.p = {**DEFAULTS, **(cfg.get("simulation") or {})}
        tf = int(cfg["bars"]["timeframe_minutes"])
        self.max_hold_bars = max(1, round(self.p["max_hold_minutes"] / tf))

        seed = seed or Seed()
        self.n_trades = seed.n_trades
        self.wins = seed.wins
        self.gross_profit = seed.gross_profit
        self.gross_loss = seed.gross_loss
        self._op_seq = seed.next_op_id

        self.pos: OpenPosition | None = None
        self.last_closed: ClosedTrade | None = None
        self.last_clips: list[Clip] = []   # pedaços gerados na última barra (p/ gravação)
        self._entries: deque[datetime] = deque()
        self._exits: deque[datetime] = deque()

    # ------------------------------------------------------------------
    def _prune(self, dq: deque[datetime], now: datetime) -> None:
        window = self.p["window_seconds"]
        while dq and (now - dq[0]).total_seconds() > window:
            dq.popleft()

    def _can_enter(self, now: datetime) -> bool:
        self._prune(self._entries, now)
        return len(self._entries) < self.p["max_entries_per_window"]

    def _can_exit(self, now: datetime) -> bool:
        self._prune(self._exits, now)
        return len(self._exits) < self.p["max_exits_per_window"]

    def _size(self, p_win: float) -> float:
        lo, hi = self.p["size_min_usd"], self.p["size_max_usd"]
        span = max(1e-9, 1.0 - self.p["entry_prob"])
        frac = min(1.0, max(0.0, (p_win - self.p["entry_prob"]) / span))
        return lo + frac * (hi - lo)

    # ------------------------------------------------------------------
    def on_bar(self, time: datetime, price: float, probs: dict[str, float]) -> dict:
        """Processa uma barra fechada; abre/reduz/zera posição simulada conforme as regras."""
        self.last_closed = None
        self.last_clips = []
        if not self.p["enabled"]:
            return self.stats()

        p_up, p_down = probs["ALTA"], probs["BAIXA"]

        if self.pos is not None:
            self.pos.bars_held += 1
            held = (time - self.pos.entry_time).total_seconds()
            opposite = p_down if self.pos.side == LONG else p_up
            if held >= self.p["min_hold_seconds"] and self._can_exit(time):
                if opposite >= self.p["exit_prob"]:
                    self._reduce(time, price, 1.0, "reversão")       # zera tudo
                elif self.pos.bars_held >= self.max_hold_bars:
                    self._reduce(time, price, 1.0, "prazo")          # zera no fim do prazo
                elif opposite >= self.p["partial_exit_prob"]:
                    self._reduce(time, price, self.p["partial_fraction"], "parcial")
        elif self._can_enter(time):
            if p_up >= self.p["entry_prob"] and (p_up - p_down) >= self.p["entry_margin"]:
                self._open(LONG, time, price, p_up)
            elif p_down >= self.p["entry_prob"] and (p_down - p_up) >= self.p["entry_margin"]:
                self._open(SHORT, time, price, p_down)

        return self.stats()

    def _open(self, side: str, time: datetime, price: float, p_win: float) -> None:
        size = self._size(p_win)
        self.pos = OpenPosition(self._op_seq, side, time, price, size, size, p_win)
        self._op_seq += 1
        self._entries.append(time)

    def _pnl_on(self, chunk: float, entry_price: float, price: float, side: str) -> float:
        ret = (price / entry_price - 1) if side == LONG else (1 - price / entry_price)
        return chunk * (ret - self.p["cost_pct"])

    def _reduce(self, time: datetime, price: float, fraction: float, reason: str) -> None:
        pos = self.pos
        chunk = pos.remaining if fraction >= 1.0 else pos.remaining * fraction
        chunk_pnl = self._pnl_on(chunk, pos.entry_price, price, pos.side)
        pos.realized_pnl += chunk_pnl
        pos.remaining -= chunk
        pos.clips += 1
        pos.last_reason = reason
        self._exits.append(time)  # cada pedaço conta como uma "saída" na janela de 6 min
        self.last_clips.append(Clip(pos.op_id, self.symbol, pos.clips, pos.side, time,
                                    pos.entry_price, price, chunk, chunk_pnl,
                                    max(0.0, pos.remaining), reason))
        if pos.remaining <= 1e-9:
            self._complete(time, price)

    def _complete(self, time: datetime, price: float) -> None:
        pos = self.pos
        won = pos.realized_pnl > 0
        reason = pos.last_reason if pos.clips == 1 else f"parcial x{pos.clips}"
        self.last_closed = ClosedTrade(pos.op_id, self.symbol, pos.side, pos.entry_time, time,
                                       pos.entry_price, price, pos.initial_size,
                                       pos.realized_pnl, won, reason, pos.clips)
        self.n_trades += 1
        self.wins += int(won)
        if pos.realized_pnl >= 0:
            self.gross_profit += pos.realized_pnl
        else:
            self.gross_loss += pos.realized_pnl
        self.pos = None

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n = self.n_trades
        winrate = self.wins / n if n else 0.0
        reached = n >= self.p["goal_trades"]
        open_realized = self.pos.realized_pnl if self.pos else 0.0
        return {
            "sim_trades": n,
            "sim_wins": self.wins,
            "sim_open": self.pos.side if self.pos else None,
            "sim_winrate": winrate,
            "sim_pnl": self.gross_profit + self.gross_loss + open_realized,
            "sim_profit": self.gross_profit,
            "sim_loss": self.gross_loss,
            "sim_goal_trades": self.p["goal_trades"],
            "sim_goal_winrate": self.p["goal_winrate"],
            "sim_goal_reached": reached,
            "sim_goal_ok": reached and winrate >= self.p["goal_winrate"],
        }
