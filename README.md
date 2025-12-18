# TinyIQA
- 一个可直接运行的 NR-IQA（无参考图像质量评价）项目，支持多模型、多数据集的训练、评估、可视化与消融实验
- 提供基于配置文件的完整流水线：Stage-1 预训练、Stage-2 MOS 回归、跨数据集评估、自动化消融与结果表生成

## 环境搭建
- Python 3.9+（建议 3.10/3.11）
- 建议使用虚拟环境
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`
- 安装依赖
  - `pip install --upgrade pip`
  - `pip install torch torchvision tqdm matplotlib numpy pyyaml pillow`
  - GPU（可选）：安装匹配 CUDA 版本的 `torch`/`torchvision`，参考官方说明

## 项目阐述
- 模型
  - `ResNetBaselineIQA`：ResNet-50 backbone，GAP(layer4)→MLP→tanh 映射到 MOS 区间
  - `StairIQA`：多层特征 1×1 对齐，layer4→3→2 的阶梯式融合，GAP→MLP 输出
  - `HyperIQA`：hyper-MLP 基于 GAP(layer4) 生成通道缩放，自适应失真模式
  - `CAQF_IQA`：layer2/3/4 多尺度融合，内容自适应权重，带/不带注意力两版
  - `TinyIQA_R18`：ResNet-18 轻量模型；`tiny_r18_kd` 模式支持对 teacher 输出的蒸馏
- 数据集
  - KonIQ-10k（主训练/评估），SPAQ、KADID-10K、AGIQA-3K（或子集）
- 训练/评估与可视化
  - tqdm 进度条显示 batch loss 与 lr
  - 每个 epoch 保存 `history.json`，并绘制 Loss 曲线与 PLCC/SRCC 曲线
  - 评估阶段生成预测 vs MOS 散点图
- 消融套件
  - 模型结构对比、训练损失对比、是否预训练、跨数据集泛化等，一键运行并汇总结果

## 目录结构（关键文件）
- `src/utils/`：配置、随机种子、指标、日志绘图、checkpoint
- `src/datasets/`：数据集加载器与失真增强
- `src/models/`：ResNet backbone 与各 IQA 头
- `src/losses/`：MSE、MAE、SRCC、RankLoss 与组合
- `src/trainers/`：Stage-1 预训练、Stage-2 回归与评估工具
- 顶层脚本：`train_stage1.py`、`train_stage2.py`、`eval_models.py`、`run_ablation_suite.py`、`make_tables.py`
- `configs/`：训练与评估/消融配置示例
- `scripts/smoke_test.py`：模型前向冒烟测试

## 数据准备
- 将图片放置到如下目录：
  - `data/koniq10k/`, `data/spaq/`, `data/kadid10k/`, `data/agiqa3k/`
- 元信息统一放在 `data/metas/` 下：
  - KonIQ JSON（示例结构）
    - `koniq_train.json`, `koniq_val.json`, `koniq_test.json`
    - 每项：`{"path": "images/xxx.jpg", "mos": 73.2}`
  - 其它数据集 CSV（示例结构）
    - `spaq_meta.csv`, `kadid10k_meta.csv`, `agiqa_meta.csv`
    - 表头至少包含：`path, mos`
- Stage-1 预训练所需的图片列表
  - `data/metas/koniq_train_list.txt`（每行一个相对路径，如 `images/xxx.jpg`）

## 运行示例
- Stage-1 合成失真预训练（可选）
  - `python train_stage1.py --image_root data/koniq10k --file_list data/metas/koniq_train_list.txt --out_ckpt outputs/stage1_pretrain.pth --epochs 1`
- Stage-2 主训练（KonIQ）
  - baseline：`python train_stage2.py --config configs/train_koniq_baseline.yaml`
  - CAQF：`python train_stage2.py --config configs/train_koniq_caqf.yaml`
  - Stair：`python train_stage2.py --config configs/train_koniq_stair.yaml`
  - Hyper：`python train_stage2.py --config configs/train_koniq_hyper.yaml`
  - Tiny：`python train_stage2.py --config configs/train_koniq_tiny.yaml`
  - Tiny-KD：先训练 baseline，得到 `outputs/koniq_baseline/resnet_baseline_best.pth`，再：
    - `python train_stage2.py --config configs/train_koniq_tiny_kd.yaml`
- 跨数据集评估（KonIQ 训练好权重后，在 SPAQ/KADID/AGIQA 上评估）
  - `python eval_models.py --config configs/eval_cross_dataset.yaml`
  - 输出：
    - `results/eval_results.json`
    - 散点图：`results/{model}_{dataset}_scatter.png`
- 自动化消融套件（顺序跑多模型与评估）
  - `python run_ablation_suite.py --suite configs/ablation_suite.yaml`
- 生成 Markdown/LaTeX 表格（用于论文/报告）
  - `python make_tables.py --results results/eval_results.json --out_md results/table.md --out_tex results/table.tex`
- 冒烟测试（不依赖数据）
  - `python scripts/smoke_test.py`

## 配置说明
- 训练配置（YAML），示例：`configs/train_koniq_baseline.yaml`
  - `mode`: `resnet_baseline`/`stair_iqa`/`hyper_iqa`/`caqf`/`caqf_no_attn`/`tiny_r18`/`tiny_r18_kd`
  - `dataset`: `koniq`/`spaq`/`kadid`/`agiqa`
  - `image_root`: 数据集图片根目录
  - `train_split`/`val_split`: 元信息文件路径（KonIQ 用 JSON，其他用 CSV）
  - `output_dir`: 输出目录（history/曲线图/最优 ckpt）
  - `epochs`/`batch_size`/`lr`/`optimizer`: 训练超参
  - `loss`: 损失设置
    - `type`: `mse`/`mae`/`mse_srcc`/`mse_rank`
    - `alpha`/`margin`: 组合损失的系数/间隔
    - KD 额外项：`kd_alpha`（仅 `tiny_r18_kd` 使用）
  - `teacher_ckpt`: 蒸馏 teacher 权重路径（`tiny_r18_kd` 模式）
  - `device`: `cuda` 或 `cpu`（会自动回退到 CPU）
  - `seed`: 随机种子
- 评估配置（YAML），示例：`configs/eval_cross_dataset.yaml`
  - `models`: 评估的模型列表
  - `datasets`: 评估的数据集列表
  - `image_roots`/`metas`: 每个数据集的图片根目录与元信息文件
  - `ckpts`: 每个模型对应的权重文件
  - `out_dir`: 评估结果输出目录

## GPU 使用与性能提示
- 代码在 CPU 上可运行以保证通用性，但速度较慢；建议使用 GPU
- 将配置中的 `device` 设置为 `cuda`，若无法使用 GPU 会自动回退到 CPU
- 适当调小 `batch_size` 与 `epochs` 以适配显存与时间预算

## 常见问题
- `torchvision` 权重下载较慢或失败
  - 可手动下载并配置环境变量 `TORCH_HOME` 指向缓存目录
- 路径不存在或元信息格式错误
  - 请确认 `image_root` 与 `train_split/val_split`/`metas` 指定的文件实际存在且格式正确
- 散点图/曲线图未生成
  - 训练/评估结束会自动保存，请检查 `output_dir` 与 `results` 目录写权限

## 参考命令速览
- 训练 baseline（KonIQ）：`python train_stage2.py --config configs/train_koniq_baseline.yaml`
- 评估跨数据集：`python eval_models.py --config configs/eval_cross_dataset.yaml`
- 跑消融套件：`python run_ablation_suite.py --suite configs/ablation_suite.yaml`
- 生成表格：`python make_tables.py --results results/eval_results.json`

