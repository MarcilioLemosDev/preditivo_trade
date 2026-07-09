"""Diagnóstico do ambiente na máquina do trader.

Roda ANTES de tudo e diz exatamente o que falta:

    python scripts/check_setup.py

Verifica: versão do Python, pacotes instalados, config.yaml, conexão
com o terminal MT5, existência de cada símbolo na corretora (com
sugestões quando o nome não bate), histórico baixado e modelos
treinados. Saída em ASCII puro para não quebrar no console do Windows.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, FAIL, WARN = "  [OK]   ", "  [FALTA]", "  [AVISO]"
problems: list[str] = []


def title(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 60 - len(text)))


def ok(text: str) -> None:
    print(OK, text)


def fail(text: str, fix: str) -> None:
    print(FAIL, text)
    problems.append(f"- {text}\n  Como resolver: {fix}")


def check_python() -> None:
    title("Python")
    ver = sys.version.split()[0]
    if sys.version_info >= (3, 10):
        ok(f"Python {ver}")
    else:
        fail(f"Python {ver} e antigo demais",
             "instale Python 3.11+ em python.org (marque 'Add python.exe to PATH')")
    bits = struct.calcsize("P") * 8
    if bits == 64:
        ok("Python 64 bits")
    else:
        fail("Python 32 bits detectado",
             "o pacote MetaTrader5 exige Python 64 bits — reinstale a versao 64-bit")


def check_packages() -> bool:
    title("Pacotes")
    required = ["pandas", "numpy", "sklearn", "lightgbm", "yaml", "joblib"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
            ok(mod)
        except ImportError:
            missing.append(mod)
            fail(f"pacote '{mod}' nao instalado", "pip install -r requirements.txt")
    try:
        import tkinter  # noqa: F401
        ok("tkinter (janelinha overlay)")
    except ImportError:
        fail("tkinter indisponivel",
             "reinstale o Python marcando o componente 'tcl/tk and IDLE' "
             "(ou use o app com --console)")
    try:
        import MetaTrader5  # noqa: F401
        ok("MetaTrader5")
    except ImportError:
        if sys.platform == "win32":
            fail("pacote MetaTrader5 nao instalado", "pip install MetaTrader5")
        else:
            print(WARN, "pacote MetaTrader5 indisponivel (normal fora do Windows; "
                        "aqui so funcionam testes e replay)")
        return False
    return not missing


def check_config():
    title("config.yaml")
    try:
        from src.config import load_config
        cfg = load_config()
        ok(f"config valida com {len(cfg['symbols'])} ativos: {', '.join(cfg['symbols'])}")
        return cfg
    except Exception as exc:  # noqa: BLE001 - diagnostico
        fail(f"config.yaml invalida: {exc}", "corrija o arquivo e rode de novo")
        return None


def check_mt5(cfg) -> None:
    import MetaTrader5 as mt5

    title("Conexao com o terminal MT5")
    kwargs = {}
    mt5_cfg = cfg.get("mt5", {})
    if mt5_cfg.get("terminal_path"):
        kwargs["path"] = mt5_cfg["terminal_path"]
    if mt5_cfg.get("login"):
        kwargs.update(login=int(mt5_cfg["login"]),
                      password=mt5_cfg.get("password", ""),
                      server=mt5_cfg.get("server", ""))
    if not mt5.initialize(**kwargs):
        fail(f"nao conectou ao terminal MT5: {mt5.last_error()}",
             "abra o terminal MT5, faca login na conta e rode de novo. "
             "Se houver mais de um MT5 instalado, preencha mt5.terminal_path "
             "no config.yaml com o caminho do terminal64.exe")
        return

    info = mt5.account_info()
    term = mt5.terminal_info()
    ok(f"conectado: conta {info.login} @ {info.server} "
       f"({'DEMO' if info.trade_mode == 0 else 'REAL'})")
    if term is not None:
        ok(f"terminal: {term.name} | historico max: {term.maxbars} barras por grafico")
        if term.maxbars < 20000:
            print(WARN, "maxbars baixo: em Ferramentas > Opcoes > Graficos, aumente "
                        "'Max. de barras no grafico' para 100000+ e reinicie o MT5")

    title("Simbolos na corretora")
    all_symbols = [s.name for s in (mt5.symbols_get() or [])]
    for name, scfg in cfg["symbols"].items():
        broker_symbol = scfg["mt5_symbol"]
        if mt5.symbol_select(broker_symbol, True):
            rates = mt5.copy_rates_from_pos(broker_symbol, mt5.TIMEFRAME_M2, 1, 100)
            n = 0 if rates is None else len(rates)
            if n > 0:
                ok(f"{name}: '{broker_symbol}' existe, {n} barras M2 de teste recebidas")
            else:
                fail(f"{name}: simbolo '{broker_symbol}' existe mas nao retornou barras M2",
                     "abra um grafico desse simbolo no MT5 para forcar o download do "
                     "historico e rode de novo")
        else:
            hint = [s for s in all_symbols if broker_symbol[:4].lower() in s.lower()][:5]
            fix = "confira o nome exato na Observacao de Mercado e ajuste config.yaml"
            if hint:
                fix += f" — parecidos nesta corretora: {', '.join(hint)}"
            fail(f"{name}: simbolo '{broker_symbol}' NAO existe nesta corretora", fix)
    mt5.shutdown()


def check_data_and_models(cfg) -> None:
    title("Historico e modelos")
    data_dir = Path(cfg["storage"]["data_dir"])
    model_dir = Path(cfg["model"]["dir"])
    for name in cfg["symbols"]:
        hist = data_dir / "history" / f"{name}.csv"
        model = model_dir / f"{name}.joblib"
        h = f"historico {'OK' if hist.exists() else 'ainda nao baixado'}"
        m = f"modelo {'treinado' if model.exists() else 'ainda nao treinado (rodara no baseline)'}"
        print(("  [OK]   " if hist.exists() else "  [ .. ] "), f"{name}: {h} | {m}")
    if not (data_dir / "history").exists():
        print(WARN, "proximo passo: python scripts/download_history.py --bars 20000")


def main() -> None:
    print("preditivo_trade — diagnostico do ambiente")
    check_python()
    has_mt5 = check_packages()
    cfg = check_config()
    if cfg is not None and has_mt5:
        check_mt5(cfg)
    if cfg is not None:
        check_data_and_models(cfg)

    title("Resumo")
    if problems:
        print(f"\n{len(problems)} pendencia(s):\n")
        print("\n".join(problems))
        sys.exit(1)
    print("\nTudo pronto. Sequencia de uso:")
    print("  1. python scripts/download_history.py --bars 20000")
    print("  2. python scripts/train.py")
    print("  3. python -m src.app --source mt5")


if __name__ == "__main__":
    main()
