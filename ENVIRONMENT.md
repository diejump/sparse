# 服务器环境

环境配置位于本目录的 `pyproject.toml`、`uv.lock` 和 `.python-version`，请一起上传到服务器。

默认环境：Python 3.10.20、PyTorch 2.0.0 + CUDA 11.8、PyG 2.5.3。
目标平台：Linux x86_64 和 Windows AMD64。GPU 训练需要支持 CUDA 11.8 的 NVIDIA 驱动。
核心依赖版本沿用本项目当前运行环境；完整间接依赖版本记录在 `uv.lock`。

## Linux 快速安装

服务器预先安装 uv，并确保可以访问 PyPI、download.pytorch.org 和 data.pyg.org。
在上传后的 `work` 目录执行：

```bash
uv sync --locked
source .venv/bin/activate
export MPLBACKEND=Agg
export PYTHONUTF8=1
python -c "import torch, torch_geometric, torch_scatter, torch_sparse, sklearn_extra, faiss; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA:', torch.version.cuda, torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
cd LoSplit-main
python run_experiments.py --dry-run
```

`uv sync --locked` 会按锁文件创建 `work/.venv`，并按 `.python-version` 选择 Python。
首次安装会下载 PyTorch 和配套包；后续同步可复用 uv 缓存。
`run_experiments.py --dry-run` 用于预览实验命令；正式运行前按 `LoSplit-main/PYTHON_RUN.md` 配置数据、权重和输出目录。

## 维护与验证

```bash
# 检查依赖声明与锁文件一致
uv lock --check

# 有意调整 pyproject.toml 后重新解析
uv lock
uv sync --locked
```

本次已完成锁文件一致性检查，以及 Linux x86_64 的 `uv sync --locked --dry-run` 安装计划检查。
服务器上的完整包安装、模块导入和 GPU 运行请通过上面的命令验证。
