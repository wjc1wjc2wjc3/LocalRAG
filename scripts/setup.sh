#!/usr/bin/env bash
# LocalRAG 环境准备（Linux / macOS）
#
#   ./scripts/setup.sh                    # 仅创建虚拟环境（核心零依赖，无需装包）
#   INSTALL_EXTRAS=1 ./scripts/setup.sh   # 额外安装可选依赖
#   NO_VENV=1 ./scripts/setup.sh          # 不建虚拟环境，直接用系统 Python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
  echo "[x] 未找到 $PY，请先安装 Python 3.9+"; exit 1
}
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "[x] 需要 Python 3.9+（当前：$("$PY" --version 2>&1)）"; exit 1
}

if [ "${NO_VENV:-0}" != "1" ]; then
  echo "[1/3] 创建虚拟环境 .venv"
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  PY=python
else
  echo "[1/3] 跳过虚拟环境（NO_VENV=1）"
fi

echo "[2/3] 升级 pip"
"$PY" -m pip install --upgrade pip -q

if [ "${INSTALL_EXTRAS:-0}" = "1" ]; then
  echo "[3/3] 安装可选依赖（语义嵌入 / PDF / HTTP API）"
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip install sentence-transformers pypdf "fastapi>=0.100" "uvicorn>=0.23"
else
  echo "[3/3] 跳过可选依赖（核心零依赖，可直接运行）"
fi

echo
echo "完成。接下来："
echo "  ./scripts/start.sh                             # 跑示例 + 测试"
echo "  ./scripts/start.sh index ./docs --acl team     # 索引文档"
echo "  ./scripts/start.sh query \"问题\" --acl team      # 检索问答"
if [ "${NO_VENV:-0}" != "1" ]; then
  echo
  echo "提示：进入虚拟环境用  source .venv/bin/activate"
fi
