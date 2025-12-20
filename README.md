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
  - KonIQ JSON（示例结构，支持多键名）
    - `koniq_train.json`, `koniq_val.json`, `koniq_test.json`
    - 每项可为：
      - `{"image": "koniq_test/10007357496.jpg", "score": 68.7285714286}`
      - 或 `{"path": "koniq_test/xxx.jpg", "mos": 73.2}`
      - MOS 键名优先级：`score` > `mos` > `MOS_zscore` > `MOS`
  - 当前仓库已提供的示例元信息文件
    - KonIQ：`data/metas/koniq_train.json`、`data/metas/koniq_test.json`
    - SPAQ：`data/metas/spaq_test.json`
    - KADID：`data/metas/kadid_test.json`
    - AGIQA：`data/metas/agiqa_test.json`
  - 可选 CSV（若你拥有官方 CSV）
    - 示例：`data/metas/koniq10k_scores_and_distributions(c1-c5...).csv`
    - 路径字段可为：`path`/`image`/`image_name`
    - MOS 字段优先级：`MOS_zscore` > `MOS` > `mos` > `score`
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
  - `image_roots`/`metas`: 每个数据集的图片根目录与元信息文件（若 JSON/CSV 内含 `koniq_test/...` 等子目录前缀，建议将 `image_root` 设为 `data`；本仓库默认使用 `*_test.json`）
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

## 统一入口：tinyiqa.py
- 查看帮助
  - `python tinyiqa.py`
- Stage-1 预训练
  - `python tinyiqa.py stage1 --image_root data/koniq10k --file_list data/metas/koniq_train_list.txt --out_ckpt outputs/stage1_pretrain.pth --epochs 1`
- Stage-2 训练
  - 指定配置：`python tinyiqa.py train --config configs/train_koniq_baseline.yaml`
  - 使用别名：
    - `python tinyiqa.py train --alias baseline`
    - `python tinyiqa.py train --alias stair`
    - `python tinyiqa.py train --alias hyper`
    - `python tinyiqa.py train --alias caqf`
    - `python tinyiqa.py train --alias caqf_no_attn`
    - `python tinyiqa.py train --alias tiny_r18`
    - `python tinyiqa.py train --alias tiny_r18_kd_stageA`
    - `python tinyiqa.py train --alias tiny_r18_kd_stageB`
    - `python tinyiqa.py train --alias tiny_r34`
    - `python tinyiqa.py train --alias tiny_r34_kd_stageA`
    - `python tinyiqa.py train --alias tiny_r34_kd_stageB`
- 跨数据集评估
  - `python tinyiqa.py eval --config configs/eval_cross_dataset.yaml`
  - 可调：`--batch_size 64 --num_workers 4`
- 消融套件（自动训练+评估）
  - `python tinyiqa.py ablation --suite configs/ablation_suite.yaml`
- 生成表格
  - `python tinyiqa.py tables --results results/eval_results.json --out_md results/table.md --out_tex results/table.tex`
- 模型统计
  - `python tinyiqa.py stats`
- 一键流水线
  - `python tinyiqa.py pipeline`

## Tiny 系列升级与两阶段 KD
- 新增模型与模式
  - `tiny_r34` / `tiny_r34_kd`：ResNet-34 轻量模型与 KD 模式
  - `tiny_r18_ms` / `tiny_r18_ms_kd`：ResNet-18 轻量多尺度版本与 KD 模式
  - 以上模式均可在配置中通过 `mode` 切换，KD 模式以 `_kd` 结尾
- 两阶段 KD（A/B）训练
  - Stage A（纯 KD，模仿 teacher）：
    - `python train_stage2.py --config configs/train_koniq_tiny_r18_kd_stageA.yaml`
    - 或 `python train_stage2.py --config configs/train_koniq_tiny_r34_kd_stageA.yaml`
  - Stage B（MOS + 轻 KD 微调）：
    - `python train_stage2.py --config configs/train_koniq_tiny_r18_kd_stageB.yaml`
    - 或 `python train_stage2.py --config configs/train_koniq_tiny_r34_kd_stageB.yaml`
  - 要求先完成 baseline 训练，确保 `kd.teacher_ckpt` 指向 `outputs/koniq_baseline/resnet_baseline_best.pth`
- KD 配置字段说明（位于训练 YAML）
  - `kd.enabled`: 是否开启 KD（布尔）
  - `kd.stage`: `"A"` 或 `"B"`（纯 KD / KD+MOS）
  - `kd.alpha`: Stage B 的轻蒸馏系数（建议 0.1–0.2）
  - `kd.teacher_ckpt`: teacher 权重路径（通常为 ResNet-50 baseline）
  - `kd.stageA_ckpt`: Stage A 输出的 ckpt，Stage B 会加载该权重继续训练
- Tiny 训练策略建议
  - `tiny_r18`: 100 epoch；`lr` 较 baseline 略小；head `Dropout(p=0.1–0.2)`
  - `tiny_r18_kd`: Stage A 20–30 epoch；Stage B 70–80 epoch；`kd.alpha` 取较小值
  - 可用 `mse_srcc` 组合损失（在 `loss.alpha` 中设置 SRCC 权重）

## 模型统计与对比
- 收集参数量与推理时间
  - `python scripts/collect_model_stats.py`
  - 输出 `results/model_stats.json`，包含 `trainable_params` 与 `avg_ms_per_image`
- 建议在相同输入尺寸下比较 baseline 与 Tiny（如 `224×224` 或 `512×512`）

## 注意事项（Tiny/KD）
- 先训练 baseline 再进行 KD（A/B），否则 teacher 权重不存在
- Stage B 必须加载 Stage A 的权重（`kd.stageA_ckpt`），否则效果不佳
- 若使用你自己的工程路径运行，请同步当前仓库的改动文件（trainer、tiny 模型、backbone 与新增配置）
- 相关性损失（SRCC）需要使用修复后的实现；若你遇到 shape 报错，可暂将 `loss.type` 设为 `mse` 先跑通，再切换回 `mse_srcc`

## 完整操作流程
- 第 0 步：环境与代码
  - 创建虚拟环境并安装依赖
  - 确认 `src/`、`configs/`、`data/`、`outputs/`、`results/` 目录存在（缺失的输出目录脚本会自动创建）
- 第 1 步：数据与元信息
  - 将图片放在 `data/` 下子目录（例如 `data/koniq_test/...`、`data/koniq_train/...`）
  - 将元信息放在 `data/metas/`：
    - KonIQ：`koniq_train.json`、`koniq_test.json`（支持键名 `image`/`path`/`image_name` 与 `score`/`mos`/`MOS_zscore`/`MOS`）
    - 其它：`spaq_test.json`、`kadid_test.json`、`agiqa_test.json`（同样支持多键名）
  - 如使用官方 CSV，将 `metas` 指向对应 `.csv` 文件即可
- 第 2 步：配置文件最小修改
  - 训练配置 YAML（示例 `configs/train_koniq_baseline.yaml`）：
    - `image_root`: 若 JSON 的 `image` 带前缀如 `koniq_test/...`，建议设为 `data`
    - `train_split`/`val_split`: 指向你实际存在的 JSON/CSV
    - `device`: `cuda` 或 `cpu`（自动回退）
    - 其余超参按需调整：`epochs`、`batch_size`、`lr`、`loss`
  - 蒸馏配置（`configs/train_koniq_tiny_kd.yaml`）：
    - 先跑 baseline，确保 `teacher_ckpt` 指向 `outputs/koniq_baseline/resnet_baseline_best.pth`
  - 评估配置（`configs/eval_cross_dataset.yaml`）：
    - `models`: 想评估的模型列表（可删减）
    - `datasets`: 想评估的数据集列表（可删减）
    - `image_roots`: 统一设为 `data` 更稳妥
    - `metas`: 指向各数据集的 `*_test.json` 或 CSV
    - `ckpts`: 各模型训练得到的最优权重路径
- 第 3 步：训练不同模型
  - baseline：`python train_stage2.py --config configs/train_koniq_baseline.yaml`
  - CAQF：`python train_stage2.py --config configs/train_koniq_caqf.yaml`
  - CAQF（无注意力）：`python train_stage2.py --config configs/train_koniq_caqf_no_attn.yaml`
  - Stair：`python train_stage2.py --config configs/train_koniq_stair.yaml`
  - Hyper：`python train_stage2.py --config configs/train_koniq_hyper.yaml`
  - Tiny：`python train_stage2.py --config configs/train_koniq_tiny.yaml`
  - Tiny-KD：`python train_stage2.py --config configs/train_koniq_tiny_kd.yaml`
  - 每次训练在对应 `outputs/{model}/` 生成 `*_history.json`、曲线图与 `*_best.pth`
- 第 4 步：跨数据集评估与可视化
  - `python eval_models.py --config configs/eval_cross_dataset.yaml`
  - 结果：`results/eval_results.json` 与散点图 `results/{model}_{dataset}_scatter.png`
- 第 5 步：自动消融与结果表
  - 一键跑：`python run_ablation_suite.py --suite configs/ablation_suite.yaml`
  - 生成表格：`python make_tables.py --results results/eval_results.json --out_md results/table.md --out_tex results/table.tex`
- 第 6 步：常见修改场景
  - 改用 CSV：仅需替换配置中的 `train_split/val_split/metas` 为 `.csv` 路径
  - 改 batch/epoch：编辑训练配置的 `batch_size`/`epochs`
  - 改损失：将 `loss.type` 改为 `mse_srcc` 或 `mse_rank`，并设置 `alpha/margin`
  - 仅评估某几个模型或数据集：在评估配置中删减对应列表项


