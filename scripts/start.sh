#!/usr/bin/env bash
# LocalRAG 运行入口（Linux / macOS）
#   无参数：跑示例 + 单元测试
#   有参数：透传给 CLI（./scripts/start.sh query "问题" --acl team）
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY="${PYTHON:-python3}"
fi

if [ $# -eq 0 ]; then
  echo "== 示例（索引 → 检索 → 溯源 → 审计）=="
  "$PY" examples/quickstart.py
  echo
  echo "== 单元测试 =="
  "$PY" -m unittest discover -s tests
else
  exec "$PY" -m localrag.cli "$@"
fi
