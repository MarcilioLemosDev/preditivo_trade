# preditivo_trade

**Modelo preditivo em tempo real para day trade** — acompanha múltiplos ativos no MetaTrader 5 em barras de 2 minutos, identifica padrões e estima, a cada barra fechada, a probabilidade do próximo movimento (subir, descer ou lateralizar), sugerindo momentos de entrada e saída numa janela compacta que não atrapalha a visão do trader.

> ⚠️ **Aviso importante**: este software é uma ferramenta de estudo e apoio à decisão. Ele **não executa ordens** e **não é recomendação de investimento**. Resultados passados não garantem resultados futuros. Toda decisão de trade é do operador humano.

---

## 1. Decisões de projeto (fechadas com o trader)

| Decisão | Escolha |
|---|---|
| Plataforma | **MetaTrader 5** — dados direto da API oficial Python, sem ler a tela |
| Ativos | NVDA, AMD, MSFT, GOOGL, Ouro (XAUUSD), BTCUSD, HK50 — configuráveis em `config.yaml` |
| Barra | 2 minutos (timeframe **M2 nativo** do MT5) |
| Interface | Janelinha compacta sempre-no-topo (overlay), uma linha por ativo |
| Execução | **Sempre humana** — o programa apenas sugere e explica |

### Por que não precisamos de prints nem de zoom

O MT5 entrega via API o histórico e o tempo real **completos, em precisão total, independente do zoom do gráfico**. O que o trader faz com zoom (micro = clareza do momento, macro = clareza do todo), o modelo faz com **janelas de análise simultâneas**: features micro (3, 5, 10 barras) capturam o momento e features macro (30, 60, 120 barras) capturam o contexto — as duas visões ao mesmo tempo, em toda barra. O trader pode dar zoom à vontade na tela dele: não afeta em nada o programa.

---

## 2. Regras formais do problema

| Conceito | Definição |
|---|---|
| **Barra** | Candle OHLC de **2 minutos**, alinhado ao relógio, fechado (a barra em formação nunca entra no pipeline). |
| **Retorno-alvo** | `r = (close_t+1 − close_t) / close_t` — fechamento da **próxima** barra contra o da atual. |
| **ALTA** | `r > +θ` |
| **BAIXA** | `r < −θ` |
| **LATERAL** | `−θ ≤ r ≤ +θ` |
| **Limiar θ** | **Adaptativo por ativo**: quantil (33%) dos \|retornos\| passados em janela móvel de 500 barras. BTC e MSFT têm volatilidades de mundos diferentes; o θ adaptativo mantém as 3 classes balanceadas em qualquer regime (verificado: 33/33/34 em teste). Modo `fixed` também disponível no `config.yaml`. |
| **Horizonte** | Sempre **1 barra à frente** (2 minutos). |
| **Gaps de sessão** | Barra seguida de gap (fim de pregão, feriado, buraco de feed) **não recebe rótulo** — o movimento seguinte dela não é um movimento de 2 minutos. Relevante para ações americanas e HK50; BTC roda 24/7. |

### Regra de ouro anti-vazamento (look-ahead)

Nenhuma feature da barra `t` usa informação posterior ao fechamento de `t`. Isso é **garantido por teste automatizado** (`tests/test_features.py::test_no_lookahead`): a feature calculada com o histórico truncado em `t` tem que ser idêntica à calculada com o dataset inteiro. Além disso, um teste de sanidade estatística treina o modelo em ruído puro (passeio aleatório) e falha se ele "vencer" o baseline — se um dia isso acontecer, é vazamento, não genialidade.

---

## 3. Stack

**Python 3.11+** (decisão fechada — sistema de tempo real, ecossistema de ML e API oficial do MT5).

| Camada | Ferramenta | Nota |
|---|---|---|
| Feed de dados | **MetaTrader5** (pacote oficial) | Windows, na máquina do terminal MT5 logado |
| Dados | **pandas** + **numpy** | |
| Indicadores | **implementados internamente** (`src/features.py`) | RSI, EMA, ATR, Bollinger etc. — sem dependência de libs de TA frágeis |
| Modelo | **LightGBM** multiclasse + calibração isotônica | baseline de frequências enquanto não há modelo treinado |
| Validação | **scikit-learn** (walk-forward via TimeSeriesSplit) | |
| UI | **tkinter** (overlay sempre-no-topo) | nativo do Python, zero instalação; modo console disponível |
| Armazenamento | **CSV append-only** (`data/`) | auditável, resistente a queda; alimenta retreino e backtest |

---

## 4. Da probabilidade ao sinal

O gerador de sinais (`src/signals.py`) só sugere quando há vantagem clara (limiares no `config.yaml`):

- **COMPRA**: `P(ALTA) ≥ 0,60` **e** `P(ALTA) − P(BAIXA) ≥ 0,25` (VENDA simétrico).
- **SAIR**: probabilidade contrária à posição sugerida ≥ 0,55.
- **FORA**: qualquer outro caso — *ficar de fora é o sinal mais comum e mais legítimo*.
- Todo sinal vem com as 3 probabilidades e o motivo em texto, e fica gravado em `data/signals/` para auditoria.

---

## 5. Arquitetura e estrutura do código

```
MT5 (M2 ao vivo) ──► DataSource ──► SymbolPipeline (1 por ativo)
                        │              features micro+macro ► modelo ► sinal
Replay (CSV) ──────────┘              │
                                       ├──► Overlay (janelinha) / console
                                       └──► data/bars + data/signals (CSV)
```

```
preditivo_trade/
├── config.yaml                  # TODOS os parâmetros de decisão
├── requirements.txt
├── src/
│   ├── config.py                # carga/validação da config
│   ├── datasource/
│   │   ├── base.py              # contrato DataSource + validação de barras
│   │   ├── mt5_source.py        # MT5 ao vivo (M2 nativo)
│   │   └── replay.py            # replay de histórico (dev, testes, simulação)
│   ├── features.py              # features micro/macro, sem look-ahead
│   ├── labeling.py              # ALTA/LATERAL/BAIXA com θ adaptativo + gaps
│   ├── model.py                 # LightGBM calibrado + baseline de frequências
│   ├── signals.py               # regras de entrada/saída
│   ├── pipeline.py              # orquestração por ativo
│   ├── recorder.py              # gravação de barras e sinais
│   ├── ui.py                    # overlay tkinter + console
│   └── app.py                   # aplicativo principal
├── scripts/
│   ├── download_history.py      # baixa histórico M2 do MT5
│   └── train.py                 # treina/valida/salva modelo por ativo
└── tests/                       # 19 testes, incluindo anti-look-ahead
```

---

## 6. Como usar

**Na máquina do trader (Windows, terminal MT5 aberto e logado):**

```bash
pip install -r requirements.txt

# 1. Conferir em config.yaml os nomes exatos dos símbolos da corretora
# 2. Baixar histórico para treino (uma vez, e depois periodicamente)
python scripts/download_history.py --bars 20000

# 3. Treinar os modelos (só salva quem bater o baseline no walk-forward)
python scripts/train.py

# 4. Acompanhar em tempo real (abre a janelinha)
python -m src.app --source mt5
```

**Desenvolvimento/simulação (qualquer sistema, sem MT5):**

```bash
python -m pytest tests/                                            # suíte completa
python -m src.app --source replay --replay-dir data/history --console --fast
```

Enquanto um ativo não tem modelo treinado (ou o modelo não bateu o baseline), o app roda com o **baseline de frequências** — que quase nunca dispara sinal, mas mantém o sistema gravando dados para o treino. Sem histórico, sem sinal: honestidade por construção.

---

## 7. Validação e honestidade estatística

1. **Walk-forward sempre**: treina no passado, testa no futuro (`TimeSeriesSplit`), nunca embaralha.
2. **Gate de produção**: `scripts/train.py` só salva o modelo se ele **bater o baseline de frequências em log-loss** no walk-forward. Modelo que não bate baseline é ruído e é descartado (comportamento verificado em teste).
3. **Calibração isotônica**: as probabilidades exibidas precisam significar o que dizem — "P(ALTA)=65%" tem que acertar ~65% das vezes.
4. **Expectativa realista**: em barras de 2 min o mercado é muito ruidoso. Um modelo bem calibrado com ~55% de acerto direcional já é valioso. Acima de ~60%, desconfie de vazamento.
5. Próxima etapa de validação: **backtest com custos** (spread, corretagem, slippage) sobre os sinais gravados.

---

## 8. Roadmap

| Fase | Entrega | Status |
|---|---|---|
| **0. Fundação** | Regras, arquitetura, README | ✅ |
| **1. Esqueleto funcional** | DataSource MT5 + replay, features, rotulagem, modelo, sinais, overlay, testes | ✅ (este commit) |
| **2. Dados reais** | Rodar `download_history.py` na máquina do trader, conferir símbolos, treinar primeiros modelos | 🔜 |
| **3. Tempo real assistido** | Rodar 1 pregão inteiro ao vivo com log completo; ajustar θ e limiares | |
| **4. Backtest com custos** | Expectância financeira dos sinais descontando spread/corretagem/slippage | |
| **5. Iteração** | SHAP nas decisões, novas features, avaliação lado a lado com o trader em paper trading | |

## 9. Pendências para a Fase 2 (precisam do trader)

1. **Nomes exatos dos símbolos na corretora** (na Observação de Mercado do MT5; alguns brokers usam sufixos: `NVDA.m`, `US_NVDA`...). Ajustar em `config.yaml`.
2. Aumentar o limite de barras do terminal, se preciso: *Ferramentas → Opções → Gráficos → Máx. de barras no gráfico*.
3. Rodar primeiro em **conta demo** para validar o fluxo ponta a ponta sem risco.
