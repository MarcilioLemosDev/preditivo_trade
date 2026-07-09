@echo off
chcp 65001 >nul
title preditivo_trade
cd /d "%~dp0"

echo.
echo  ============================================
echo   preditivo_trade - acompanhamento em tempo real
echo  ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo  Python nao encontrado nesta maquina.
  echo.
  echo  Instale uma unica vez: https://www.python.org/downloads/
  echo  IMPORTANTE: na primeira tela do instalador, marque a caixinha
  echo  "Add python.exe to PATH" antes de clicar em Install Now.
  echo.
  echo  Depois e so dar dois cliques neste arquivo de novo.
  echo.
  pause
  exit /b 1
)

python -c "import pandas, numpy, sklearn, lightgbm, yaml, joblib" >nul 2>nul
if errorlevel 1 (
  echo  Primeira execucao: preparando o programa, aguarde uns minutos...
  echo.
  python -m pip install --quiet --disable-pip-version-warning -r requirements.txt
  if errorlevel 1 (
    echo.
    echo  Nao consegui instalar os componentes. Verifique a internet e tente de novo.
    pause
    exit /b 1
  )
)

echo  Deixe o MetaTrader 5 aberto e logado, com o grafico do ativo na tela.
echo  A janelinha vai abrir e acompanhar sozinha os graficos abertos.
echo  Para encerrar, feche a janelinha (ou esta janela preta).
echo.

python -m src.app --source mt5

echo.
pause
