@echo off
setlocal

set "DATA=%~1"
if "%DATA%"=="" set "DATA=test.json"

set "OUT=%~2"
if "%OUT%"=="" (
    python -m src.solve --data "%DATA%"
) else (
    python -m src.solve --data "%DATA%" --out "%OUT%"
)
