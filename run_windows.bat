@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set NO_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set no_proxy=

set "PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/"
set "PIP_TRUSTED_HOST=mirrors.aliyun.com"
set "PIP_OPTS=-i %PIP_INDEX_URL% --trusted-host %PIP_TRUSTED_HOST%"

set "PYTHON_CMD="
echo 正在检测 Python...
call :try_python python
if not defined PYTHON_CMD call :try_python py
if not defined PYTHON_CMD call :try_python python3

if "%PYTHON_CMD%"=="" (
  echo 未找到 Python。请先安装 Python 3.10 或更高版本。
  echo 如果已经安装 Python，请确认命令行中可以运行 python、py 或 python3。
  pause
  exit /b 1
)

echo 使用 Python 命令: %PYTHON_CMD%
%PYTHON_CMD% -c "import sys; print('Python ' + sys.version.split()[0]); raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)"
if errorlevel 1 (
  echo 当前 Python 版本低于 3.10，建议安装 Python 3.10 或更高版本。
  pause
  exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo 正在创建本地虚拟环境...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo 虚拟环境创建失败。请确认 %PYTHON_CMD% 可正常运行并自带 venv 模块。
    pause
    exit /b 1
  )
)

if not exist "%VENV_PY%" (
  echo 虚拟环境创建失败：没有找到 %VENV_PY%
  echo 如果之前运行失败留下了损坏的 .venv 文件夹，请删除 .venv 后重新运行本脚本。
  pause
  exit /b 1
)

echo 使用 pip 镜像源: %PIP_INDEX_URL%
"%VENV_PY%" -m pip install %PIP_OPTS% --upgrade pip
"%VENV_PY%" -m pip install %PIP_OPTS% -r requirements.txt

echo 正在启动数字孪生课堂演示网页...
echo 如果浏览器没有自动打开，请访问: http://localhost:8501
"%VENV_PY%" -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
pause
exit /b 0

:try_python
echo 尝试: %~1
%~1 -c "import sys; raise SystemExit(0)" >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=%~1"
)
exit /b 0
