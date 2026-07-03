#!/bin/bash
set -e

cd "$(dirname "$0")"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
PIP_TRUSTED_HOST="mirrors.aliyun.com"
export PIP_INDEX_URL PIP_TRUSTED_HOST

PYTHON_CMD=""
echo "正在检测 Python..."
for candidate in python3 python py; do
  echo "尝试: $candidate"
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1; then
    PYTHON_CMD="$candidate"
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "未找到 Python 3.10 或更高版本。请先安装 Python 3.10+。"
  exit 1
fi

echo "使用 Python 命令: $PYTHON_CMD"
"$PYTHON_CMD" -c "import sys; print('Python ' + sys.version.split()[0])"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "正在创建本地虚拟环境..."
  "$PYTHON_CMD" -m venv .venv
fi

if [ ! -x "$VENV_PY" ]; then
  echo "虚拟环境创建失败：没有找到 $VENV_PY"
  echo "如果之前运行失败留下了损坏的 .venv 文件夹，请删除 .venv 后重新运行本脚本。"
  exit 1
fi

echo "使用 pip 镜像源: $PIP_INDEX_URL"
"$VENV_PY" -m pip install -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" --upgrade pip
"$VENV_PY" -m pip install -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" -r requirements.txt
"$VENV_PY" -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
