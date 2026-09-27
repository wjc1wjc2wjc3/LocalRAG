@echo off
rem LocalRAG 运行入口（Windows）
rem   无参数：跑示例 + 单元测试
rem   有参数：透传给 CLI（scripts\start.bat query "问题" --acl team）
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=py -3"
)

if "%~1"=="" (
  echo == 示例（索引 → 检索 → 溯源 → 审计）==
  %PY% examples\quickstart.py
  echo.
  echo == 单元测试 ==
  %PY% -m unittest discover -s tests
) else (
  %PY% -m localrag.cli %*
)
endlocal
