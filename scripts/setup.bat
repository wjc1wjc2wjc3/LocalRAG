@echo off
rem LocalRAG 环境准备（Windows）
rem   scripts\setup.bat                          仅创建虚拟环境（核心零依赖）
rem   set INSTALL_EXTRAS=1 ^&^& scripts\setup.bat 额外安装可选依赖
rem   set NO_VENV=1 ^&^& scripts\setup.bat        不建虚拟环境
setlocal
cd /d "%~dp0.."

set "PY=py -3"
where py >nul 2>nul
if errorlevel 1 set "PY=python"

%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)"
if errorlevel 1 (
  echo [x] 需要 Python 3.9+
  exit /b 1
)

if "%NO_VENV%"=="1" goto :skipvenv
echo [1/3] 创建虚拟环境 .venv
%PY% -m venv .venv
if errorlevel 1 (
  echo [x] 创建虚拟环境失败
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
goto :aftervenv
:skipvenv
echo [1/3] 跳过虚拟环境（NO_VENV=1）
:aftervenv

echo [2/3] 升级 pip
%PY% -m pip install --upgrade pip -q

if "%INSTALL_EXTRAS%"=="1" (
  echo [3/3] 安装可选依赖（语义嵌入 / PDF / HTTP API）
  %PY% -m pip install -r requirements.txt
  %PY% -m pip install sentence-transformers pypdf "fastapi>=0.100" "uvicorn>=0.23"
) else (
  echo [3/3] 跳过可选依赖（核心零依赖，可直接运行）
)

echo.
echo 完成。接下来：
echo   scripts\start.bat
echo   scripts\start.bat index docs --acl team
echo   scripts\start.bat query "问题" --acl team
endlocal
