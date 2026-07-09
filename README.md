# preditivo_trade

**Modelo preditivo em tempo real para day trade** — acompanha o gráfico de uma ação em barras de 2 minutos, identifica padrões e estima, a cada barra, a probabilidade do próximo movimento (subir, descer ou lateralizar), sugerindo momentos de entrada e saída.

> ⚠️ **Aviso importante**: este software é uma ferramenta de estudo e apoio à decisão. Ele **não executa ordens** e **não é recomendação de investimento**. Resultados passados não garantem resultados futuros. Toda decisão de trade é responsabilidade do operador.

---

## 1. Escopo do MVP

**O que o MVP faz:**

- Acompanha o gráfico em **tempo real** (barras de 2 minutos).
- Reconstrói a série de candles (OHLC) a partir da fonte de dados.
- Calcula features técnicas a cada barra fechada.
- Estima a probabilidade do **próximo movimento de 2 minutos**: `P(subir)`, `P(descer)`, `P(lateralizar)`.
- Emite **sugestões de entrada e saída** quando as probabilidades cruzam os limiares definidos (seção 4).
- Registra tudo em log para auditoria e backtest posterior.

**O que o MVP não faz (por decisão de projeto):**

- ❌ Não envia ordens à corretora (sem automação de execução).
- ❌ Não faz gestão de carteira ou sizing de posição.
- ❌ Não opera múltiplos ativos simultaneamente (1 ativo por instância).

---

## 2. Regras formais do problema

Para o modelo ser treinável e auditável, as definições precisam ser exatas:

| Conceito | Definição |
|---|---|
| **Barra** | Candle OHLC consolidado em janela fixa de **2 minutos**, alinhada ao relógio (ex.: 10:00:00–10:01:59). |
| **Barra verde** | `close > open` |
| **Barra vermelha** | `close < open` |
| **Retorno da barra** | `r = (close_t+1 − close_t) / close_t` — variação do fechamento da **próxima** barra em relação ao fechamento da atual. |
| **Subir (classe ALTA)** | `r > +θ` |
| **Descer (classe BAIXA)** | `r < −θ` |
| **Lateralizar (classe LATERAL)** | `−θ ≤ r ≤ +θ` |
| **Limiar θ (theta)** | Definido por ativo, com base na volatilidade e no tick mínimo. Ponto de partida sugerido: `θ = 0,05%` do preço **ou** 1× o spread médio — o que for maior. Será calibrado com dados reais. |
| **Horizonte de previsão** | Sempre **1 barra à frente** (2 minutos). Previsões multi-barra ficam fora do MVP. |
| **Momento da previsão** | A previsão para a barra `t+1` é emitida **imediatamente após o fechamento da barra `t`**, usando apenas informação disponível até `close_t`. |

### Regra de ouro anti-vazamento (look-ahead)

Nenhuma feature pode usar informação posterior ao fechamento da barra atual. Isso inclui: não usar `high/low/close` da barra em formação, não normalizar com estatísticas calculadas sobre o dataset inteiro (usar janelas móveis), e validar sempre com **walk-forward** (treina no passado, testa no futuro — nunca embaralhar temporalmente).

---

## 3. Linguagem e stack recomendadas

**Decisão: Python.** R é excelente para análise estatística exploratória, mas Python vence com folga para este projeto porque o sistema é de **tempo real** (loop de captura, pipeline de eventos, UI ao vivo) e o ecossistema de ML/visão computacional/streaming em Python é muito mais maduro para produção.

| Camada | Ferramenta | Por quê |
|---|---|---|
| Linguagem | **Python 3.11+** | Ecossistema, tempo real, deploy |
| Dados/série temporal | **pandas** + **numpy** | Padrão de mercado para OHLC |
| Indicadores técnicos | **pandas-ta** | RSI, EMA, MACD, ATR, Bandas de Bollinger etc. sem dependência de compilação (alternativa ao TA-Lib) |
| Modelo baseline | **LightGBM** (via scikit-learn API) | Estado da arte para dados tabulares, rápido para treinar/inferir, lida bem com poucas amostras iniciais |
| Validação/calibração | **scikit-learn** | Walk-forward split, calibração de probabilidade (isotônica/Platt) |
| Captura de tela (se necessário) | **mss** + **OpenCV** | Screenshot de alta frequência + detecção de candles por cor/forma |
| Dashboard em tempo real | **Streamlit** + **Plotly** | UI ao vivo com gráfico de candles, probabilidades e sinais, com pouquíssimo código |
| Logs/armazenamento | **SQLite** + **Parquet** | Zero infraestrutura, suficiente para 1 ativo em barras de 2 min |
| Backtest | **vectorbt** (fase 2) | Backtest vetorizado rápido das regras de sinal |

Modelos sequenciais (LSTM/Transformer via PyTorch) ficam para a fase 3 — só fazem sentido depois que o baseline tabular estiver medido, senão não temos referência para saber se agregam valor.

---

## 4. Da probabilidade ao sinal (regras de entrada/saída)

O modelo entrega probabilidades; o **gerador de sinais** converte em sugestão apenas quando há vantagem estatística acima dos custos:

- **Sugerir COMPRA**: `P(ALTA) ≥ 0,60` **e** `P(ALTA) − P(BAIXA) ≥ 0,25`.
- **Sugerir VENDA/SHORT**: simétrico com `P(BAIXA)`.
- **Sugerir SAÍDA de posição comprada**: `P(BAIXA) ≥ 0,55` **ou** stop/alvo atingido na barra.
- **Sem sinal (ficar de fora)**: qualquer outro caso — inclusive quando `P(LATERAL)` domina. *Ficar de fora é um sinal válido e será o mais frequente.*
- Todo sinal exibe: probabilidades das 3 classes, features que mais pesaram na decisão (via importâncias/SHAP) e horário exato.

Esses limiares são **parâmetros de configuração** (`config.yaml`), não constantes no código, e serão recalibrados com o backtest.

---

## 5. Arquitetura do MVP

```
┌─────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Fonte de dados  │ →  │ Agregador de │ →  │  Motor de     │ →  │  Gerador de  │
│  (DataSource)    │    │ barras 2min  │    │  features +   │    │  sinais +    │
│                  │    │ (OHLC)       │    │  modelo       │    │  dashboard   │
└─────────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
                                                    ↓
                                          log SQLite/Parquet
                                          (auditoria + retreino)
```

### 5.1 Fonte de dados — decisão crítica do projeto

O pedido original é "acompanhar a tela". Há dois caminhos, e a arquitetura suporta ambos via uma interface única `DataSource`:

**Opção A — Captura de tela (visão computacional):** `mss` tira screenshots da região do gráfico e OpenCV detecta os candles por cor/posição, convertendo pixels em preços via calibração dos eixos.
- ✅ Funciona com qualquer plataforma gráfica, sem depender de API.
- ❌ Frágil: qualquer mudança de zoom, layout ou tema quebra a leitura; precisão limitada a pixels; sem volume confiável; exige recalibração frequente.

**Opção B — Feed de dados direto (recomendada):** obter cotações diretamente da fonte e montar as barras de 2 min nós mesmos. Caminhos comuns no mercado brasileiro:
- **MetaTrader 5** (pacote oficial `MetaTrader5` para Python) — a maioria das corretoras BR oferece MT5; dá acesso a tick e OHLC em tempo real, de graça.
- **Profit (Nelogica) via RTD/DDE** para Excel → Python.
- Provedores de market data (Cedro, dados B3 via corretora).

**Recomendação séria de quem já viu isso dar errado:** começar direto pela Opção B com MetaTrader 5 se a corretora do trader oferecer. A Opção A consome semanas de engenharia de visão computacional para entregar dados piores do que a API entrega em um dia. A captura de tela fica como *fallback* para plataformas fechadas. **Primeira pergunta a responder antes de codar: qual plataforma o trader usa?**

### 5.2 Estrutura de diretórios proposta

```
preditivo_trade/
├── README.md
├── config.yaml              # ativo, θ, limiares de sinal, região da tela etc.
├── requirements.txt
├── src/
│   ├── datasource/          # interface DataSource + implementações
│   │   ├── base.py          #   (mt5, screen_capture, replay)
│   ├── bars.py              # agregador de barras 2min
│   ├── features.py          # engenharia de features (só passado!)
│   ├── labeling.py          # rotulagem ALTA/BAIXA/LATERAL com θ
│   ├── model.py             # treino, calibração, inferência
│   ├── signals.py           # regras de entrada/saída
│   └── app.py               # dashboard Streamlit em tempo real
├── notebooks/               # exploração e análise (fora do caminho de produção)
├── data/                    # parquet de barras históricas (git-ignored)
└── tests/
```

---

## 6. Modelagem e validação

1. **Features iniciais** (todas com janela olhando só para trás): retornos das últimas N barras, corpo/pavios do candle, sequência de cores, RSI, EMAs (9/21) e distância do preço a elas, ATR (volatilidade), posição nas Bandas de Bollinger, volume relativo, hora do dia (abertura/almoço/fechamento têm dinâmicas distintas), distância de máximas/mínimas do dia.
2. **Baseline obrigatório antes de qualquer modelo**: prever "sempre LATERAL" e prever "repete a cor da última barra". O modelo só presta se bater esses dois de forma consistente.
3. **Modelo**: LightGBM multiclasse → probabilidades calibradas (calibração isotônica em janela de validação).
4. **Validação**: walk-forward com janela deslizante (ex.: treina 30 dias, testa 5, desliza). Métricas: **log-loss**, **Brier score**, acurácia balanceada e — a que importa de verdade — **expectância financeira dos sinais no backtest descontando custos** (corretagem, emolumentos, slippage de 1 tick).
5. **Honestidade estatística**: mercado em 2 min é muito ruidoso; um modelo com 55% de acerto direcional bem calibrado já é bom. Desconfie de qualquer resultado acima de ~60% — quase sempre é vazamento de dados.

---

## 7. Roadmap

| Fase | Entrega | Critério de conclusão |
|---|---|---|
| **0. Fundação** | Este README, `config.yaml`, esqueleto do projeto | Regras validadas com o trader |
| **1. Dados** | `DataSource` funcionando (MT5 ou captura de tela) + agregador de barras + gravação histórica | 5+ dias de barras 2min gravadas e conferidas contra a plataforma |
| **2. Baseline** | Features + rotulagem + LightGBM + walk-forward + backtest com custos | Métricas honestas reportadas, batendo os baselines ingênuos |
| **3. Tempo real** | Dashboard Streamlit ao vivo com probabilidades e sinais | Rodar 1 pregão inteiro sem travar, com log completo |
| **4. Iteração** | Calibração de θ e limiares, SHAP, novas features, avaliação lado a lado com o trader | Trader validando os sinais em paper trading |

---

## 8. Perguntas em aberto (responder antes da Fase 1)

1. **Qual plataforma gráfica o trader usa?** (Profit, MT5, TradingView, homebroker?) → define a Opção A vs B.
2. **Qual ativo/contrato?** (WIN, WDO, ação específica?) → define θ, custos e horário de pregão.
3. Existe histórico de barras 2min disponível para acelerar o treino inicial, ou começamos gravando do zero?
