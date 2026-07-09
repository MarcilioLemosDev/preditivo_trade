# preditivo_trade

**Modelo preditivo em tempo real para day trade** — uma janelinha compacta que enxerga sozinha quais gráficos estão abertos no MetaTrader 5 e, a cada barra de 2 minutos, mostra a probabilidade do próximo movimento (subir, descer ou lateralizar) e sugere momentos de entrada e saída. Quem executa é sempre o humano.

> ⚠️ **Aviso importante**: este software é uma ferramenta de estudo e apoio à decisão. Ele **não executa ordens** e **não é recomendação de investimento**. Resultados passados não garantem resultados futuros. Toda decisão de trade é do operador humano.

---

## Como abrir o programa (guia do trader — sem parte técnica)

**O que você precisa uma única vez (5 minutos):**

1. Instale o Python: entre em [python.org/downloads](https://www.python.org/downloads/), baixe e execute o instalador. **Na primeira tela, marque a caixinha "Add python.exe to PATH"** e clique em *Install Now*.
2. Baixe esta pasta do projeto para o computador (botão verde **Code → Download ZIP** no GitHub, e descompacte onde quiser — por exemplo na Área de Trabalho).

**No dia a dia (sempre igual):**

1. Abra o **MetaTrader 5** e deixe logado, com o gráfico do ativo que você vai operar na tela (pode abrir vários gráficos).
2. Dê **dois cliques no arquivo `INICIAR.bat`** dentro da pasta do projeto.
3. Pronto. A janelinha abre e faz o resto sozinha:
   - **Descobre quais gráficos você abriu** no MT5 e passa a acompanhá-los. Abriu um gráfico novo? Ele entra na janelinha. Fechou? Ele sai.
   - Na primeira vez que vê um ativo, aparece **"aprendendo"** — o programa está estudando o histórico em segundo plano (alguns minutos). Quando termina, vira **"pronto"**.
   - A cada 2 minutos, atualiza as probabilidades de **ALTA / LATERAL / BAIXA** e a sugestão: **COMPRA**, **VENDA**, **SAIR**, **MANTER** ou **FORA**.
4. Para encerrar, feche a janelinha.

Na primeira execução o `INICIAR.bat` instala os componentes sozinho (precisa de internet e leva alguns minutos). Nas seguintes, abre direto.

**Coisas boas de saber:**

- **Zoom não importa.** O programa recebe os dados direto do MT5, não da imagem da tela. Dê zoom à vontade.
- **"FORA" é o sinal mais comum — e está certo.** Em janelas de 2 minutos, na maior parte do tempo não há vantagem estatística. O programa só sugere entrada quando enxerga vantagem clara.
- **Quer que a previsão olhe mais longe?** Abra o arquivo `config.yaml` no Bloco de Notas e mude `horizon_bars`: `1` = próximos 2 min, `2` = próximos 4 min, `3` = próximos 6 min... Salve e abra o programa de novo — ele aprende sozinho na nova janela.
- Se algo der errado, a janela preta que abre junto mostra a mensagem — mande ela para o suporte (nós 🙂).

*(Hoje o programa abre por esse duplo clique; um executável único `.exe` que dispensa até a instalação do Python está no roadmap — é um passo de empacotamento, não de funcionalidade.)*

---

## 1. Decisões de projeto

| Decisão | Escolha |
|---|---|
| Plataforma | **MetaTrader 5** — dados direto da API oficial Python, sem ler imagem da tela |
| Ativos | **Detecção automática dos gráficos abertos** no MT5 (NVDA, AMD, MSFT, GOOGL, XAUUSD, BTCUSD, HK50, ou qualquer outro que o trader abrir) |
| Barra | 2 minutos (timeframe **M2 nativo** do MT5) |
| Horizonte do insight | **Ajustável** em `config.yaml` (`prediction.horizon_bars`): h barras de 2 min à frente |
| Treino | **Automático em segundo plano** na primeira vez que um ativo é visto |
| Interface | Janelinha compacta sempre-no-topo (overlay), uma linha por ativo |
| Execução | **Sempre humana** — o programa apenas sugere e explica |

### Como o programa "vê a tela" sem ler pixels

A detecção dos gráficos abertos lê os **títulos das janelas** do MT5 via API do Windows (as janelas de gráfico chamam-se `SIMBOLO,TEMPO`, ex.: `XAUUSD,M2`). Já os **dados** (preços, barras) vêm da API oficial do MT5 — completos e em precisão total, independentes de zoom. A visão "macro e micro" do trader é reproduzida por **janelas de análise simultâneas**: features micro (3, 5, 10 barras) capturam o momento; features macro (30, 60, 120 barras) capturam o contexto.

---

## 2. Regras formais do problema

| Conceito | Definição |
|---|---|
| **Barra** | Candle OHLC de **2 minutos**, alinhado ao relógio, fechado (a barra em formação nunca entra no pipeline). |
| **Horizonte h** | `prediction.horizon_bars` (1 = 2 min, 2 = 4 min, ...). Cada horizonte tem **modelo próprio** (`models/ATIVO_h{h}.joblib`) — prever 2 min e 6 min são tarefas diferentes. |
| **Retorno-alvo** | `r = (close_t+h − close_t) / close_t` — fechamento h barras à frente contra o da barra atual. |
| **ALTA** | `r > +θ` |
| **BAIXA** | `r < −θ` |
| **LATERAL** | `−θ ≤ r ≤ +θ` |
| **Limiar θ** | **Adaptativo por ativo e por horizonte**: quantil (33%) dos \|retornos de h barras\| passados em janela móvel de 500 barras. Mantém as 3 classes balanceadas em qualquer regime de volatilidade (verificado: 33/33/34 em teste). Modo `fixed` disponível. |
| **Gaps de sessão** | Barra cujo alvo cruza gap (fim de pregão, feriado, buraco de feed) **não recebe rótulo**. Relevante para ações americanas e HK50; BTC roda 24/7. |

### Regra de ouro anti-vazamento (look-ahead)

Nenhuma feature da barra `t` usa informação posterior ao fechamento de `t`. **Garantido por teste automatizado** (`tests/test_features.py::test_no_lookahead`). Um segundo guarda-corpo treina o modelo em ruído puro (passeio aleatório) e falha se ele "vencer" o baseline — se acontecer, é vazamento, não genialidade.

---

## 3. Stack

**Python 3.11+** (sistema de tempo real, ecossistema de ML e API oficial do MT5).

| Camada | Ferramenta | Nota |
|---|---|---|
| Feed de dados | **MetaTrader5** (pacote oficial) | Windows, na máquina do terminal MT5 logado |
| Detecção de gráficos | **ctypes/WinAPI** (títulos de janela) | zero dependências externas |
| Dados | **pandas** + **numpy** | |
| Indicadores | **implementados internamente** (`src/features.py`) | RSI, EMA, ATR, Bollinger — sem libs de TA frágeis |
| Modelo | **LightGBM** multiclasse + calibração isotônica | baseline de frequências enquanto não há modelo aprovado |
| Validação | **scikit-learn** (walk-forward via TimeSeriesSplit) | |
| UI | **tkinter** (overlay sempre-no-topo) | nativo do Python; modo console disponível |
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
MT5 ──► chart_watch (janelas abertas) ──► app: 1 SymbolPipeline por gráfico
  │                                             │
  └──► MT5Source (barras M2 fechadas) ──────────┤  features micro+macro
                                                 │  ► modelo (ou baseline)
Replay (CSV) ► mesmo pipeline, p/ dev/simulação  │  ► sinal + explicação
                                                 ├──► Overlay / console
       trainer (thread em 2º plano,              └──► data/bars + data/signals
       treina ao ver ativo novo) ◄───────────────┘
```

```
preditivo_trade/
├── INICIAR.bat                  # duplo clique do trader (instala e abre)
├── config.yaml                  # parâmetros; trader só mexe em horizon_bars
├── requirements.txt
├── src/
│   ├── config.py                # carga/validação da config
│   ├── chart_watch.py           # detecção dos gráficos abertos no MT5
│   ├── datasource/
│   │   ├── base.py              # contrato DataSource + validação de barras
│   │   ├── mt5_source.py        # MT5 ao vivo (M2, símbolos dinâmicos)
│   │   └── replay.py            # replay de histórico (dev, testes, simulação)
│   ├── features.py              # features micro/macro, sem look-ahead
│   ├── labeling.py              # ALTA/LATERAL/BAIXA, θ adaptativo, horizonte h
│   ├── model.py                 # LightGBM calibrado + baseline de frequências
│   ├── trainer.py               # treino (usado pelo app em 2º plano e pelo script)
│   ├── signals.py               # regras de entrada/saída
│   ├── pipeline.py              # orquestração por ativo
│   ├── recorder.py              # gravação de barras e sinais
│   ├── ui.py                    # overlay tkinter (linhas dinâmicas) + console
│   └── app.py                   # aplicativo principal (detecção + auto-treino)
├── scripts/
│   ├── check_setup.py           # diagnóstico do ambiente
│   ├── download_history.py      # baixa histórico M2 em lote (uso avançado)
│   └── train.py                 # retreino em lote com métricas (uso avançado)
└── tests/                       # 25 testes, incluindo anti-look-ahead
```

---

## 6. Uso técnico (desenvolvedores)

```bash
pip install -r requirements.txt
python -m pytest tests/                    # suíte completa (roda em qualquer SO)
python scripts/check_setup.py              # diagnóstico do ambiente

# ao vivo (Windows + MT5 aberto) — o que o INICIAR.bat chama:
python -m src.app --source mt5

# simulação com histórico gravado (qualquer SO):
python -m src.app --source replay --replay-dir data/history --console --fast

# fluxo em lote (opcional; o app treina sozinho em 2º plano):
python scripts/download_history.py --bars 20000
python scripts/train.py
```

---

## 7. Validação e honestidade estatística

1. **Walk-forward sempre**: treina no passado, testa no futuro (`TimeSeriesSplit`), nunca embaralha.
2. **Gate de produção**: o modelo (treinado pelo app ou pelo script) só entra em uso se **bater o baseline de frequências em log-loss** no walk-forward. Modelo que não bate é descartado e o ativo segue no modo básico (comportamento verificado em teste).
3. **Calibração isotônica**: "P(ALTA)=65%" tem que acertar ~65% das vezes.
4. **Expectativa realista**: em barras de 2 min o mercado é muito ruidoso. ~55% de acerto direcional bem calibrado já é valioso; acima de ~60%, desconfie de vazamento.
5. Próxima etapa: **backtest com custos** (spread, corretagem, slippage) sobre os sinais gravados em `data/signals/`.

---

## 8. Roadmap

| Fase | Entrega | Status |
|---|---|---|
| **0. Fundação** | Regras, arquitetura, README | ✅ |
| **1. Esqueleto funcional** | DataSource MT5 + replay, features, rotulagem, modelo, sinais, overlay, testes | ✅ |
| **2. Experiência do usuário final** | Detecção automática de gráficos, auto-treino em 2º plano, horizonte ajustável, `INICIAR.bat` | ✅ (este commit) |
| **3. Primeiro pregão real** | Rodar um dia inteiro na máquina do trader; validar detecção, sinais e logs | 🔜 |
| **4. Backtest com custos** | Expectância financeira dos sinais descontando spread/corretagem/slippage | |
| **5. Iteração** | SHAP nas decisões, novas features, avaliação lado a lado com o trader; empacotar `.exe` (PyInstaller) | |
