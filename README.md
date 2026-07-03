# 数字孪生课堂演示包

这是一个独立的 Streamlit 网页演示包，内置 100 个脱敏课堂演示画像。学生下载后可以本地运行，不需要下载完整 Twin-2K-500 数据集。

## 功能

- 查看和比较 Twin-2K-500 数字孪生画像
- 单题模拟选择题和开放题回答
- 批量评估数字孪生预测表现
- 生成需求曲线、管理策略沙盘、市场细分雷达和市场调研结果
- 支持无 API Key 的 Mock 演示模式，也支持接入 DeepSeek API

## 一键运行

macOS:

1. 解压压缩包。
2. 双击 `run_mac.command`。
3. 脚本会自动寻找 `python3`、`python` 或 `py`。
4. 第一次运行会自动创建 `.venv`，之后安装依赖和启动网页都会使用 `.venv` 里的 Python。
5. 打开网页：`http://localhost:8501`

Windows:

1. 解压压缩包。
2. 双击 `run_windows.bat`。
3. 脚本会自动寻找 `python`；如果找不到，会依次改用 `py` 或 `python3`。
4. 第一次运行会自动创建 `.venv`，之后安装依赖和启动网页都会使用 `.venv` 里的 Python。
5. 打开网页：`http://localhost:8501`

Linux:

```bash
chmod +x run_linux.sh
./run_linux.sh
```

Linux 脚本同样会自动寻找 `python3`、`python` 或 `py`，创建 `.venv` 后会固定使用 `.venv/bin/python`。

## API Key

不填写 API Key 也可以运行，网页会进入 Mock 演示模式。

如果要使用真实 DeepSeek API，请在网页左侧栏的 `DeepSeek API Key` 输入框中粘贴自己的 Key。不要把 Key 写进代码文件，也不要把包含 Key 的截图发给别人。

## 下载依赖

启动脚本默认使用阿里 PyPI 镜像源：`https://mirrors.aliyun.com/pypi/simple/`，国内网络下载会更快。如果学校网络无法访问该镜像，可以在启动脚本中把 `PIP_INDEX_URL` 改成其他镜像源。

## 包含内容

- `streamlit_app.py`：课堂网页
- `twin_core.py`：数据读取和 LLM 调用逻辑
- `data/twin_2k_500_local/`：100 个画像的本地数据
- `tmp/zh_translation_cache.json`：中文翻译缓存
- `requirements.txt`：运行依赖
- `run_mac.command` / `run_windows.bat` / `run_linux.sh`：启动脚本

## 许可证与数据归属

本仓库的应用代码和启动脚本使用 MIT License，详见 `LICENSE`。

内置的 Twin-2K-500 示例数据以及由其生成的中文翻译缓存不属于 MIT 授权范围；它们按 Twin-2K-500 数据集的 CC BY 4.0 许可和归属要求使用。详见 `DATA_LICENSE.md` 和 `NOTICE`。

使用或再分发内置数据时，请引用：

```bibtex
@dataset{twin2k500,
  author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
  title     = {Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People},
  year      = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://arxiv.org/abs/2505.17479}}
}
```

## 常见问题

如果端口被占用，可以先关闭其他正在运行的 Streamlit 窗口，或者在脚本中把 `8501` 改成其他端口。

如果之前运行失败过，文件夹里可能留下损坏的 `.venv`。可以先删除 `.venv` 文件夹，再重新运行对应系统的启动脚本。

如果 macOS 提示无法打开脚本，可以右键点击 `run_mac.command`，选择“打开”；或在终端中运行：

```bash
chmod +x run_mac.command
./run_mac.command
```
