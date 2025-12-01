# 基于 ResNet-50 的 NR-IQA 项目（两阶段预训练 + 对比实验）

## 项目概述
- 无参考图像质量评价（NR-IQA）完整实现：基于 ResNet-50 的 CAQF-IQA 主模型，包含多尺度内容自适应融合与可选空间注意力。
- 两阶段训练流程：
  - Stage-1：仅用 KonIQ-10k 原图构造合成失真分类预训练（不引入外部数据）。
  - Stage-2：在 KonIQ MOS 标签上进行回归微调。
- 统一协议对比：ResNet-50 baseline / Stair-IQA（简化封装）/ HyperIQA（简化封装）。
- 指标：PLCC、SRCC；可选跨数据集零样本评估（SPAQ / KADID-10K / AGIQA-3K）。

## 主要特性
- ResNet-50 多尺度特征抽取（`layer2/layer3/layer4`）。
- CAQF 融合：自适应权重生成，尺度融合，支持空间注意力，统一 MOS 映射到 `[0, 100]`。
- 配置驱动：所有路径与超参数由 YAML 管理，避免硬编码。
- 模块化：datasets / models / trainers / utils 清晰分层，便于复现实验与扩展。

## 环境配置
- Python 3.8+（建议 3.8/3.9/3.10）
- PyTorch 1.12+（建议安装匹配 CUDA 的版本）
- 依赖：`torchvision`, `numpy`, `Pillow`, `PyYAML`

安装示例：
- 使用 Conda（推荐）
```
conda create -n nriqa python=3.9 -y
conda activate nriqa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 如无 CUDA 可改为 CPU 版本
pip install numpy Pillow PyYAML
```
- 使用 Pip（CPU 或已装好 CUDA 环境）
```
pip install torch torchvision
pip install numpy Pillow PyYAML
```

## 数据准备
- KonIQ-10k：按如下目录组织，并准备标签 CSV：
```
Data/
  koniq_train/
  koniq_test/
  metas/            # 例如 koniq_mos.csv
  spaq_test/
  kadid_test/
  agiqa_test/
```
  - `metas/koniq10k_scores_and_distributions.csv` 示例（首行字段）：
    `image_name,c1,c2,c3,c4,c5,c_total,MOS,SD,MOS_zscore`
    实际训练默认读取 `MOS_zscore` 字段（0–100 刻度）；若改用 `MOS`（1–5 刻度），请把 `out_range` 改为 `[1.0, 5.0]`。
```
image_name,mos
12345.jpg,56.78
67890.jpg,41.23
```
  - 其他数据集仅用于评估（只读）。

## 目录结构
```
configs/
  train_stage1_distortion_cls.yaml
  train_stage2_mos_regression.yaml
  eval_cross_dataset.yaml
src/
  datasets/
    koniq_dataset.py
    distortion_augment.py
  models/
    backbones/resnet_backbone.py
    iqa_heads/caqf_iqa.py
    iqa_heads/resnet_baseline.py
    stair_iqa_wrapper.py
    hyper_iqa_wrapper.py
  losses/regression_losses.py
  trainers/
    trainer_stage1_distortion_cls.py
    trainer_stage2_mos_regression.py
    trainer_eval_only.py
  utils/
    metrics.py
    logging_utils.py
    config_utils.py
train_stage1.py
train_stage2.py
eval_models.py
make_tables.py
```

## 配置说明
- `configs/train_stage1_distortion_cls.yaml`
  - `dataset_root`: KonIQ 训练图片根目录（如 `Data/koniq_train`）
  - `csv_path`: 可选标签 CSV，不需要 MOS 时可留空
  - `batch_size`, `epochs`, `learning_rate`
  - `num_classes`: 合成失真类别总数（含 clean 与不同强度）
  - `freeze_stem_layer1`: 是否冻结 `stem+layer1`
  - `image_size`: 训练输入尺寸
- `configs/train_stage2_mos_regression.yaml`
  - `dataset_root`: KonIQ 训练图片根目录（如 `Data/koniq_train`）
  - `mos_csv_path`: MOS 标签 CSV（如 `Data/metas/koniq10k_scores_and_distributions.csv`）
  - `mos_field`: 选择使用的字段（默认 `MOS_zscore`，可设为 `MOS`）
  - `out_range`: 模型输出范围；`MOS_zscore` 用 `[0,100]`，`MOS` 用 `[1,5]`
  - `batch_size`, `epochs`, `learning_rate`, `image_size`
  - `mode`: `ours_stage1_pretrained` / `ours_no_pretrain` / `resnet_baseline`
  - `stage1_ckpt`: Stage-1 最优权重路径（`ours_stage1_pretrained` 时使用）
  - `lambda_srcc`: SRCC 损失权重
- `configs/eval_cross_dataset.yaml`
  - `datasets`: 各数据集路径（主要使用 KonIQ）
  - `models`: 评估的模型名与对应 checkpoint 路径

## 快速开始
- Stage-1：失真分类预训练（使用 `Data/koniq_train`）
```
python train_stage1.py --config configs/train_stage1_distortion_cls.yaml
```
输出：`checkpoints/stage1_best.pth`

- Stage-2：MOS 回归训练（三种模式）（使用 `Data/koniq_train` + `metas/koniq_mos.csv`）
  - 我们（加载 Stage-1）：
```
python train_stage2.py --config configs/train_stage2_mos_regression.yaml
# 确保配置中 mode=ours_stage1_pretrained 且 stage1_ckpt 指向 checkpoints/stage1_best.pth
```
  - 我们（不加载 Stage-1）：修改配置 `mode: ours_no_pretrain` 后运行同上。
  - ResNet-50 baseline：修改配置 `mode: resnet_baseline` 后运行同上。
输出：`checkpoints/stage2_best_<mode>.pth`
- 统一评估（KonIQ 测试集）
```
python eval_models.py --config configs/eval_cross_dataset.yaml --root Data/koniq_test --csv Data/metas/koniq10k_scores_and_distributions.csv --list_json Data/metas/koniq_test.json --mos_field MOS
```
输出：`results/eval_results.json`

- 生成报告表格（Markdown）
```
python make_tables.py --results results/eval_results.json
```
输出：标准 Markdown 表格片段，可直接复制到论文/报告。

## 模型说明
- Backbone：`src/models/backbones/resnet_backbone.py`
  - 加载 ImageNet 预训练的 ResNet-50，拆分 `stem, layer1-4`，前向返回 `f2, f3, f4`。
- CAQF-IQA：`src/models/iqa_heads/caqf_iqa.py`
  - 将 `f2, f3, f4` 对齐至同尺度，用 `1x1` 卷积投影到统一通道数 `C`。
  - `GAP` 后由 MLP 生成三维权重，`softmax` 归一化；按权重融合三尺度特征。
  - 可选空间注意力：`Conv(C,1)`+`sigmoid` 得到注意力图进行加权。
  - `GAP` 输出向量经回归头（`C -> C//2 -> 1`），`tanh` 映射到 `[0,100]`。
- Baseline：`src/models/iqa_heads/resnet_baseline.py`
  - 仅使用 `f4`，`GAP -> MLP -> tanh` 映射到 `[0,100]`。
- Stair-IQA（简化）：`src/models/stair_iqa_wrapper.py`
  - 三尺度投影后 `GAP` 拼接，MLP 回归，统一输出范围。
- HyperIQA（简化）：`src/models/hyper_iqa_wrapper.py`
  - 用超网络根据 `f4` 的 `GAP` 向量生成通道仿射参数 `(gamma, beta)` 调制 `f4`，再回归输出。

## 日志与检查点
- 日志：`checkpoints/stage1_log.csv`、`checkpoints/stage2_log.csv`（包含 loss 与验证指标）。
- 检查点：`checkpoints/stage1_best.pth`、`checkpoints/stage2_best_<mode>.pth`。

## 自检与调试
- 模型前向烟测：
```
python scripts/smoke_test_models.py
```
- 失真函数烟测：
```
python scripts/smoke_test_datasets.py
```
- 导入自检（Windows）：
```
python -c "import importlib,os,sys;sys.path.append(os.getcwd());importlib.import_module('src.trainers.trainer_stage1_distortion_cls');importlib.import_module('src.trainers.trainer_stage2_mos_regression');importlib.import_module('src.trainers.trainer_eval_only');print('import_ok')"
```

## 常见问题
- 包导入与 `__init__.py`：本项目使用隐式命名空间包（PEP 420），不需要 `__init__.py`。从项目根目录运行命令即可正常导入。
- CUDA 与加速：若安装了匹配 CUDA 的 PyTorch，训练将自动使用 GPU；若仅 CPU，训练速度会变慢。
- CSV 字段名：若标签列名与示例不同（如 `MOS`/`label`），已做兼容处理，但建议统一为 `mos`。

## 许可与引用
- 数据集版权与协议请遵循原始发布方（KonIQ-10k、SPAQ、KADID-10K、AGIQA-3K）。
- 参考实现基于 ResNet-50 与典型 NR-IQA 思路，Stair/HyperIQA 为简化封装以便于统一对比与教学使用。
