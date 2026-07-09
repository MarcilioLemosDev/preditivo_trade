"""Parsing dos títulos de janela do MT5 (a parte testável em qualquer SO)."""

from src.chart_watch import parse_active_chart, parse_chart_title


def test_parses_child_chart_titles():
    assert parse_chart_title("XAUUSD,M2") == "XAUUSD"
    assert parse_chart_title("NVDA,M15") == "NVDA"
    assert parse_chart_title("BTCUSD,H1") == "BTCUSD"
    assert parse_chart_title("HK50,Daily") == "HK50"
    assert parse_chart_title("US500,Weekly") == "US500"
    assert parse_chart_title("PETR4,Monthly") == "PETR4"


def test_parses_broker_suffixes():
    assert parse_chart_title("NVDA.m,M2") == "NVDA.m"
    assert parse_chart_title("US_NVDA,M2") == "US_NVDA"
    assert parse_chart_title("GOLD#,M30") == "GOLD#"


def test_rejects_non_chart_titles():
    assert parse_chart_title("Sem título - Bloco de Notas") is None
    assert parse_chart_title("MetaTrader 5") is None
    assert parse_chart_title("Observação de Mercado") is None
    assert parse_chart_title("") is None
    # vírgula sem timeframe válido não é gráfico
    assert parse_chart_title("relatorio,final") is None


def test_parses_active_chart_from_main_title():
    title = "10005 - MetaQuotes-Demo: Conta Demo - Hedge - [XAUUSD,M2] - MetaTrader 5"
    assert parse_active_chart(title) == "XAUUSD"
    assert parse_active_chart("MetaTrader 5 - sem grafico ativo") is None
