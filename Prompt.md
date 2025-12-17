#### 角色设定（给本地代码 Agent 的系统指令）

你是一个偏工程与研究结合的代码生成 Agent，需要**从零或在已有基础上生成一整套 NR-IQA（无参考图像质量评价）项目代码**，支持多模型、多数据集对比，提供完整的训练 & 评估 & 可视化 & 消融实验流水线。

目标不是只写几段 demo，而是生成一个**可以直接运行、结构清晰、方便写实验报告**的完整项目。

---

#### 一、总体目标

1. **实现并比较多种 IQA 模型**（尽量贴近原文架构）：
   - ResNet-50 baseline（单阶段回归）
   - Stair-IQA（多层阶梯式特征融合，尽量复现原文结构）
   - HyperIQA（hyper-network 结构，尽量复现原文结构）
   - 我们自定义的多尺度模型 CAQF_IQA
   - TinyIQA_R18（ResNet-18 轻量模型）及其蒸馏版本 TinyIQA_R18_KD

2. **支持多个数据集上的指标对比**：
   - KonIQ-10k（主训练 + 主评估数据集）
   - SPAQ
   - KADID-10K
   - AGIQA-3K（或者本地已有的 AGIQA 子集）
   - 要支持：
     - 单数据集训练 + 单数据集测试
     - KonIQ 训练 + 其他数据集**零微调测试**（跨数据集泛化实验）

3. **提供详细训练过程可视化**：
   - 批次级：使用 `tqdm` 显示训练进度条（loss 等信息）
   - epoch 级：
     - 保存训练/验证 loss 曲线图
     - 保存验证集 PLCC / SRCC 曲线图
   - 测试阶段：
     - 预测 vs MOS 散点图
   - 所有图用 `matplotlib` 绘制并保存为 `.png`

4. **设计并自动运行一系列消融实验**：
   - 结构消融：
     - ResNet-50 baseline vs CAQF_IQA vs Stair-IQA vs HyperIQA
     - ResNet-50 baseline vs TinyIQA_R18 vs TinyIQA_R18_KD
     - CAQF 有/无注意力（ours_no_attn）
   - 训练策略消融：
     - 仅 MSE vs MSE+SRCC（相关性 loss） vs 加 RankLoss
     - 有/无 Stage-1 预训练（合成失真分类）对 CAQF 的影响
     - 有/无数据增强 & 一致性正则（增强两视图输出约束）
   - 泛化消融：
     - 在 KonIQ 上训练不同模型，然后在 SPAQ / KADID / AGIQA 上直接测试，看性能掉落情况

5. **生成面向论文/报告的结果文件**：
   - 将所有实验结果（PLCC、SRCC、RMSE、MAE 等）保存为统一的 JSON/CSV
   - 提供一个脚本自动生成 Markdown / LaTeX 表格片段，方便写报告时直接粘贴。

---

#### 二、项目结构要求

在当前项目根目录下自动创建如下结构（如已存在则合并）：

- `configs/`
  - `train_koniq_baseline.yaml`
  - `train_koniq_stair.yaml`
  - `train_koniq_hyper.yaml`
  - `train_koniq_caqf.yaml`
  - `train_koniq_tiny.yaml`
  - `train_koniq_tiny_kd.yaml`
  - `train_koniq_stage1_pretrain.yaml`（合成失真预训练）
  - `eval_cross_dataset.yaml`
  - 若干消融用配置，如：
    - `train_koniq_caqf_no_attn.yaml`
    - `train_koniq_baseline_mse_srcc.yaml` 等
- `data/`
  - `koniq10k/`       （图像文件根目录）
  - `spaq/`
  - `kadid10k/`
  - `agiqa3k/`
  - `metas/`          （各种 CSV / JSON 列表）
    - `koniq10k_scores_and_distributions.csv`
    - `koniq_train.json`, `koniq_val.json`, `koniq_test.json`
    - `spaq_meta.csv`, `kadid10k_meta.csv`, `agiqa_meta.csv`（结构可通过代码注释说明）
- `src/`
  - `datasets/`
    - `koniq_dataset.py`
    - `spaq_dataset.py`
    - `kadid_dataset.py`
    - `agiqa_dataset.py`
    - `distortion_augment.py`（Stage-1 合成失真构造）
  - `models/`
    - `backbones/`
      - `resnet_backbone.py`（实现 ResNet-50 / ResNet-18 backbone）
    - `iqa_heads/`
      - `resnet_baseline.py`（ResNetBaselineIQA）
      - `caqf_iqa.py`       （CAQF_IQA + no_attn 变体）
      - `stair_iqa.py`      （Stair-IQA 结构，尽量贴原文）
      - `hyper_iqa.py`      （HyperIQA 结构，尽量贴原文）
      - `tiny_iqa.py`       （TinyIQA_R18 & TinyIQA_R18_KD）
  - `losses/`
    - `regression_losses.py`（MSE、MAE、SRCC loss、RankLoss 等）
  - `trainers/`
    - `trainer_stage1_pretrain.py`      （失真分类/排序预训练）
    - `trainer_stage2_regression.py`    （KonIQ MOS 回归）
    - `trainer_eval_only.py`            （测试&可视化）
  - `utils/`
    - `metrics.py`        （PLCC, SRCC, RMSE, MAE 等）
    - `logging_utils.py`  （历史记录 & Matplotlib 绘图）
    - `config_utils.py`   （YAML 配置解析）
    - `seed_utils.py`     （统一设置随机种子）
    - `checkpoint_utils.py`（保存/加载 ckpt，支持 teacher/student）
- 顶层脚本：
  - `train_stage1.py`      （运行 Stage-1 预训练）
  - `train_stage2.py`      （KonIQ 主训练）
  - `eval_models.py`       （多模型多数据集评估）
  - `run_ablation_suite.py`（自动跑一组预设消融实验）
  - `make_tables.py`       （生成 Markdown/LaTeX 表格）

要求：所有脚本都可以通过命令行参数 + 配置文件方式运行。

---

#### 三、模型实现细节（尽量复现原文架构）

##### 3.1 ResNet-50 Baseline（现有最强 baseline）

- Backbone：ImageNet 预训练 ResNet-50（torchvision）
  - 使用 `conv1+bn1+relu+maxpool + layer1~4`
  - 移除原 fc 层
- Head：
  - GAP(layer4) → `Linear(2048, hidden)` → ReLU → `Linear(hidden, 1)`
  - 用 `tanh` 映射到 MOS 区间（例如 [0, 100]）
- Loss：
  - 默认 MSE，可在 config 中切换为 MSE+SRCC、MSE+RankLoss 等（用于消融）
- 作为 teacher 提供给 TinyIQA 蒸馏使用。

##### 3.2 Stair-IQA（参考原文结构，尽量还原）

实现一个类 `StairIQA`，要求：

- 基于 ResNet-50 backbone；
- 特征层级：
  - 使用多层特征（例如 layer1~4 或 layer2~4），每层经过 1×1 conv 对齐维度；
  - 采用“阶梯式”从高层逐步融合回低层：
    - 例如从 layer4 开始，插值/上采样到 layer3 尺度，与 layer3 concat/加和，再继续到 layer2…；
- 每一步融合后可以加简单的 conv+BN+ReLU；
- 最终输出一个 feature map，经 GAP + MLP 得到 MOS；
- 尽量遵循 Stair-IQA 论文中的结构思路，例如：
  - 多尺度特征融合由 coarse-to-fine 逐步完成；
  - 保持参数量与论文同量级（可以用注释写明与原文可能存在微小差异）。

##### 3.3 HyperIQA（hyper network 结构）

实现一个类 `HyperIQA`，要求：

- Backbone：ResNet-50；
- 核心思想：
  - hyper-network 生成一些自适应参数（例如通道权重 / BN gamma-beta / attention 权重）；
  - 这些参数用来调节主干网络中某些层的响应，使网络对失真模式更自适应；
- 简化但保持原意：
  - 可以把最后若干 conv block 的某些通道的缩放系数由 hyper-MLP 产出；
  - hyper-MLP 的输入可以是全球特征或者中间层 GAP 特征；
- 最终还是 GAP → MLP 回归 MOS。

##### 3.4 CAQF_IQA（我们自己的多尺度模型）

实现一个 `CAQF_IQA` 类，包含：

- Backbone：ResNet-50；
- 多尺度特征：
  - 提取 layer2, layer3, layer4；
  - 通过 `adaptive_avg_pool` 将它们对齐到 layer4 空间尺寸；
  - 使用 1×1 conv 把通道统一到 C（比如 256）得到 `f2a, f3a, f4a`；
- 内容自适应权重（CAQF）：
  - 对每个 `fka` 做 GAP 得到 `g2, g3, g4`；
  - concat 成 `[g2; g3; g4]` 送入 MLP 生成三维权重向量 `w`；
  - softmax 后得到 `w2,w3,w4`，做加权和：
    - $F_q = w_2 f2a + w_3 f3a + w_4 f4a$
- 注意力（带/不带两种）：
  - 带注意力版：`A = sigmoid(Conv2d(C, 1, kernel=1)(F_q))`，再 `F_q * A`；
  - no_attn 版：直接跳过 attention；
- Head：
  - GAP → MLP(C → C/2 → 1) → MOS。

##### 3.5 TinyIQA_R18 + KD（轻量模型）

实现 `TinyIQA_R18` 与蒸馏版本能力：

- `TinyIQA_R18`：
  - Backbone：ResNet-18（ImageNet 预训练）
  - GAP(layer4) → MLP(512→256→1)，可选 dropout；
- 在训练阶段支持两种模式：
  - `tiny_r18`：仅对 MOS 回归；
  - `tiny_r18_kd`：对 MOS 回归 + 对 teacher（ResNet-50 baseline）的输出做 KD；
- KD loss：
  - `L_MOS = MSE(y_student, y_true)`
  - `L_KD  = MSE(y_student, y_teacher)`
  - `L = L_MOS + α * L_KD`（α 在 config 中设置，如 0.5）。

---

#### 四、训练与评估流水线

##### 4.1 通用训练脚本 `train_stage2.py`

要求：

- 使用 YAML config 指定：
  - `mode`：`resnet_baseline`, `stair_iqa`, `hyper_iqa`, `caqf`, `caqf_no_attn`, `tiny_r18`, `tiny_r18_kd` 等；
  - 数据集路径、训练/验证划分文件；
  - loss 配置（是否加 SRCC、RankLoss 等）；
  - 优化器、学习率、epoch 数。
- 根据 `mode`：
  - 构建对应模型；
  - 如果是 KD 模式，加载 teacher ckpt；
  - 如果是 Stage-1 预训练相关实验，对 CAQF model 加载预训练权重。
- 使用 `tqdm` 包裹 dataloader，显示：
  - 当前 batch idx / 总 batch 数；
  - 当前 batch loss；
- 每个 epoch：
  - 在 train loader 上训练，返回 avg train loss；
  - 在 val loader 上评估，计算：
    - val_loss（MSE 或组合 loss）
    - val_PLCC / val_SRCC（用 `metrics.py` 实现）
  - 保存到 `history` 中，并写入 JSON/CSV；
  - 根据 val_SRCC 或 val_PLCC 选择最优 ckpt（存到 `checkpoints/{model_name}_best.pth`）。

##### 4.2 通用评估脚本 `eval_models.py`

要求：

- 接受参数：
  - `--config configs/eval_cross_dataset.yaml`
  - 指定：
    - 要评估的模型列表（字符串列表）：`["resnet_baseline", "stair_iqa", "hyper_iqa", "caqf", "caqf_no_attn", "tiny_r18", "tiny_r18_kd"]`
    - 每个模型的 ckpt 路径；
    - 要评估的数据集列表：`["koniq", "spaq", "kadid", "agiqa"]`；
- 对每个 `(model, dataset)` 组合：
  - 加载对应 Dataset 和 DataLoader；
  - 加载模型 ckpt，设置 eval 模式；
  - 运行前向，收集所有预测 & MOS；
  - 计算 PLCC, SRCC, RMSE, MAE；
- 最终把结果汇总为一个 JSON，例如：

```json
{
  "resnet_baseline": {
    "koniq": {"plcc": ..., "srcc": ..., "rmse": ..., "mae": ...},
    "spaq":  {...},
    "kadid": {...},
    "agiqa": {...}
  },
  "stair_iqa": { ... },
  ...
}
```

并保存为 `results/eval_results.json`。

---

#### 五、训练过程可视化（tqdm + 图像）

##### 5.1 训练日志 & tqdm

- 在训练循环中使用 `tqdm`：

```python
from tqdm import tqdm

for batch in tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False):
    ...
    tqdm_bar.set_postfix(loss=..., lr=...)
```

- 对验证阶段也可用 tqdm，但可以只显示进度，不必太详细。

##### 5.2 `logging_utils.py`：绘图工具

实现函数：

```python
def save_history(history: list[dict], out_path: str):
    # history 中每个 dict 至少包含 epoch, train_loss, val_loss, val_plcc, val_srcc
    # 保存为 JSON

def plot_training_curves(history: list[dict], out_dir: str, title_prefix: str = ""):
    """
    生成两张图：
    1. loss_curve.png：epoch vs train/val loss
    2. corr_curve.png：epoch vs val_plcc / val_srcc
    """
```

范例：

```python
import matplotlib.pyplot as plt
import os

def plot_training_curves(history, out_dir, title_prefix=""):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_plcc = [h["val_plcc"] for h in history]
    val_srcc = [h["val_srcc"] for h in history]

    # loss
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curves")
    plt.legend()
    plt.grid(True)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_loss_curve.png"))
    plt.close()

    # corr
    plt.figure()
    plt.plot(epochs, val_plcc, label="val_PLCC")
    plt.plot(epochs, val_srcc, label="val_SRCC")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.title(f"{title_prefix} PLCC/SRCC Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_corr_curve.png"))
    plt.close()
```

##### 5.3 预测 vs MOS 散点图

在 `trainer_eval_only.py` 中实现：

```python
def plot_pred_vs_mos(y_true, y_pred, out_path: str, title: str = ""):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="y=x")
    plt.xlabel("MOS (ground truth)")
    plt.ylabel("Predicted MOS")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
```

在 eval 时调用：

```python
scatter_path = os.path.join(out_dir, f"{model_name}_{dataset_name}_scatter.png")
plot_pred_vs_mos(y_true_list, y_pred_list, scatter_path,
                 title=f"{model_name} on {dataset_name}")
```

---

#### 六、消融实验设计与自动运行脚本

实现 `run_ablation_suite.py`，自动顺序跑以下实验（每个实验对应一个 YAML 配置文件）：

1. **结构消融 1：多模型对比（KonIQ 上）**
   - `resnet_baseline`
   - `stair_iqa`
   - `hyper_iqa`
   - `caqf`
   - `caqf_no_attn`

2. **结构消融 2：Tiny vs Baseline（KonIQ 上）**
   - `resnet_baseline`
   - `tiny_r18`
   - `tiny_r18_kd`

3. **损失函数消融（以 baseline 为例）**
   - `baseline_mse`
   - `baseline_mse_srcc`
   - `baseline_mse_rank`

4. **预训练消融（CAQF）**
   - `caqf_no_pretrain`（仅 Stage-2）
   - `caqf_stage1_pretrain`（先 Stage-1 失真分类，再 Stage-2 MOS 回归）

5. **泛化能力消融**
   - 对上面若干模型，在 `eval_models.py` 中：
     - 使用 KonIQ 训练好的权重；
     - 在 SPAQ / KADID / AGIQA 上直接测试；
     - 比较跨数据集性能掉落。

`run_ablation_suite.py` 的职责：

- 读取一个简单的 JSON/YAML 列表，依次：
  - 调用 `train_stage2.py` 训练模型；
  - 调用 `eval_models.py` 跨数据集评估；
- 最后整合所有 `eval_results.json` 为一个总表，用于生成最终 Markdown/LaTeX 表。

---

#### 七、实现风格与工程要求

1. 使用的主要第三方库：
   - `torch`, `torchvision`
   - `tqdm`
   - `matplotlib`
   - `numpy`
   - `pyyaml`
2. 所有模块都要有简短 docstring，说明其用途和与实验的关系。
3. 保持代码风格统一（例如 PEP8，函数命名统一用 snake_case）。
4. 提供若干 README 说明（可选）：
   - 顶层 `README.md`：简述项目结构与运行命令示例；
   - `docs/experiments.md`：列出所有实验和对应配置文件名（可以由代码生成）。

---

#### 八、你需要完成的事情（给 Agent 的 checklist）

1. **按照上述目录结构生成/补全项目代码文件**。
2. **完整实现所有模型类**：ResNetBaselineIQA、StairIQA、HyperIQA、CAQF_IQA(+no_attn)、TinyIQA_R18(+KD)。
3. 实现 `datasets/` 中各数据集的 Dataset 类，支持根据 CSV/JSON 元数据读取图像和 MOS。
4. 实现 `metrics.py`（PLCC、SRCC、RMSE、MAE）。
5. 实现 `regression_losses.py`（MSE、MAE、SRCC loss、RankLoss 等）。
6. 实现训练脚本 `train_stage1.py`、`train_stage2.py`，支持 tqdm 与 history 记录。
7. 实现评估及可视化脚本 `trainer_eval_only.py` 与 `eval_models.py`。
8. 实现 `logging_utils.py` 绘图函数，生成 loss & PLCC/SRCC 曲线以及散点图。
9. 实现 `run_ablation_suite.py` 和 `make_tables.py`，自动跑消融并生成结果表（JSON+Markdown/LaTeX）。
10. 保证所有脚本至少在 CPU 环境下可跑（速度慢没关系），并在代码内部用注释说明 GPU 使用方法。

请严格按照以上要求生成项目代码，使我只需按说明顺序执行脚本，就能获得：
- 各模型在 KonIQ / SPAQ / KADID / AGIQA 上的对比指标；
- 完整的训练曲线图和测试散点图；
- 系统性的消融实验结果和可直接用于写报告的表格。
