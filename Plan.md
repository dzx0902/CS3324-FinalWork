#### 角色设定（给本地代码 Agent 的系统指令）

你是一个负责**改进 TinyIQA 轻量模型**的代码 Agent，要在**现有 NR-IQA 项目代码基础上**做一轮升级，目标是：

- 保持/进一步降低模型复杂度（参数/FLOPs/推理时间）；
- 明显提升 Tiny 系列模型在 KonIQ / SPAQ / KADID / AGIQA 上的表现；
- 特别是让 Tiny 模型的 **PLCC/SRCC 明显逼近 ResNet-50 baseline**；
- 通过 **更合理的蒸馏（KD）和针对小模型的训练策略** 达成，而不是随便堆结构。

请不要推翻整个项目结构，而是在现有结构上**增量修改**和扩展。

---

### 一、现有前提（你可以假定已存在）

项目中已经有：

1. 模型：
   - `ResNetBaselineIQA`（ResNet-50 baseline，当前最强 teacher）
   - `TinyIQA_R18`（基于 ResNet-18 的 Tiny 模型）
   - `TinyIQA_R18_KD`（一版带 KD 的 Tiny，当前效果不佳）
   - `StairIQA`、`HyperIQA`、`CAQF_IQA` 等其他模型

2. 训练/评估脚本：
   - `train_stage2.py`：KonIQ MOS 回归主训练脚本（支持 `mode` 字段）
   - `eval_models.py`：在 KonIQ / SPAQ / KADID / AGIQA 计算 PLCC、SRCC、RMSE、MAE
   - `metrics.py`：已实现 PLCC / SRCC 等
   - `logging_utils.py`：已有或部分实现训练记录和绘图（可以扩展）

3. 结果示例（多模型多数据集对比的表，已验证 pipeline 可用）。

你的任务是在此基础上，**专门针对 Tiny 线进行系统性增强**。

---

### 二、总体目标拆解

你需要完成以下几件事：

1. **重构 TinyIQA 的蒸馏逻辑**：改成“两阶段 KD + 轻权重蒸馏”，让 KD 真正提升性能。
2. **增加更强但仍然轻量的 Tiny 变体**（如 ResNet-34 或轻量多尺度 Tiny），并用 KD 训练。
3. **为 Tiny 模型定制训练策略**：更合适的 epoch、学习率、loss 组合（MSE + SRCC）。
4. **重新跑 Tiny 系列实验**，并与 baseline 在多个数据集上对比。
5. **补充参数量/时间对比和表格输出**，方便最后写报告。

下面是具体实现要求。

---

### 三、重构 TinyIQA 的蒸馏（KD）

#### 3.1 定义更清晰的 Tiny 模式

在 `train_stage2.py`（或等价入口）中，确保支持以下 `mode`：

- `"tiny_r18"`：ResNet-18 Tiny，**无 KD**，只对 MOS 回归；
- `"tiny_r18_kd"`：ResNet-18 Tiny，**两阶段 KD+MOS**；
- （可选）`"tiny_r34"` / `"tiny_r34_kd"`：ResNet-34 Tiny 及 KD 版本。

你需要有一个统一的构建函数，例如：

```python
def build_model_and_teacher(cfg, device):
    """
    根据 cfg.mode 返回 (model, teacher)。
    - 对于不需要 KD 的模式，teacher=None。
    - 对于 KD 模式，加载并冻结 ResNetBaselineIQA 作为 teacher。
    """
```

要求：

- 当 `mode` 以 `_kd` 结尾时：
  - 构建 student：对应 Tiny 模型（R18 或 R34）；
  - 构建 teacher：`ResNetBaselineIQA`，加载 `cfg.teacher.ckpt_path`；
  - `teacher.eval()`，所有参数 `requires_grad=False`；
  - teacher **不加入 optimizer**。

#### 3.2 两阶段 KD 训练流程

你需要给 Tiny KD 模式实现“两阶段训练”：

##### Stage A：纯 KD 预训练

- 目标：让 Tiny 学会模仿 teacher 的输出；
- 损失：

  ```python
  L = MSE(y_student, y_teacher)
  ```

- 不使用 MOS 标签（或者 MOS loss 权重设置为 0）；
- epoch 数可以较少，比如 20–30（通过 config 控制）；
- 训练结束后保存 ckpt（例如 `checkpoints/tiny_r18_kd_stageA.pth`）。

##### Stage B：MOS + 轻 KD 微调

- 从 Stage A ckpt 加载 Tiny 权重；
- 重新创建 optimizer；
- 损失组合：

  ```python
  L_mos = MSE(y_student, y_true)
  L_kd  = MSE(y_student, y_teacher)
  L     = L_mos + alpha * L_kd
  ```

- `alpha` 在配置中设置，例如 `0.1` 或 `0.2`，不要过大；
- epoch 数与 baseline 相近或略多（如 80–100）。

这两个阶段可以由同一个脚本完成（通过 config 中的标记），也可以拆成两个脚本：

- `train_tiny_kd_stageA.py`
- `train_tiny_kd_stageB.py`

但建议是在现有 `train_stage2.py` 中，通过配置字段控制，例如：

```yaml
mode: tiny_r18_kd

kd:
  enabled: true
  stage: "A"   # or "B"
  alpha: 0.1
  teacher_ckpt: "checkpoints/resnet_baseline_best.pth"
  stageA_ckpt: "checkpoints/tiny_r18_kd_stageA.pth"
```

在代码中根据 `kd.stage` 决定使用纯 KD 还是 KD+MOS。

---

### 四、增强 Tiny 模型结构（在仍然轻量的前提下）

#### 4.1 增加 TinyIQA_R34

在 `src/models/backbones/resnet_backbone.py` 中：

- 在已有的 `ResNetBackbone50` 与 `ResNetBackbone18` 基础上，补充：

```python
class ResNetBackbone34(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_dim = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
```

在 `src/models/iqa_heads/tiny_iqa.py` 中定义：

- `TinyIQA_R34`：结构类似 `TinyIQA_R18`，只是 backbone 换成 ResNet-34，head 可稍微宽一点（如 512→512→1）；
- `TinyIQA_R34` 也支持一个 `dropout_p` 参数，默认 0.1–0.2 用于 regularization。

#### 4.2 可选：轻量多尺度 Tiny

如果你有精力，可以再定义一个 `TinyIQA_R18_MS`：

- backbone：ResNet-18；
- 使用 `layer3` + `layer4`：
  - 对 `layer3` 输出做上采样到 `layer4` 尺度；
  - 各自用 1×1 conv 压到 256 通道；
  - 用一个非常小的 MLP 产生两个 scalar 权重（softmax），组合两层特征；
- GAP → MLP 输出 MOS。

这个模型也可以纳入 KD 方案中（teacher 仍然是 ResNet-50 baseline）。

---

### 五、专门为 Tiny 设计训练策略

为 Tiny 系列单独准备 YAML 配置（例如 `configs/train_koniq_tiny_r18.yaml`、`train_koniq_tiny_r18_kd_stageA.yaml` 等），与 baseline 区分开：

建议 Tiny 的训练配置：

- 训练 epoch：
  - `tiny_r18`:   100 epoch
  - `tiny_r18_kd`：
    - Stage A: 20–30 epoch
    - Stage B: 70–80 epoch
- 学习率：
  - 初始 lr 略小于 baseline，或者用 cosine decay；
- 正则：
  - head 上加 Dropout(p=0.1–0.2)；
  - Data augmentation 稍微强一些（增加 color jitter、gamma 调整等）；

同时，为 Tiny 的部分实验增加 SRCC loss 版本：

- 在 `regression_losses.py` 中确保有 `srcc_loss(pred, target)`；
- 配置中添加：

```yaml
loss:
  type: "mse_srcc"
  mse_weight: 1.0
  srcc_weight: 0.2
```

- 在训练 loop 中根据配置组合损失：

```python
if cfg.loss.type == "mse":
    loss = mse_loss(pred, target)
elif cfg.loss.type == "mse_srcc":
    loss = mse_loss(pred, target) + cfg.loss.srcc_weight * srcc_loss(pred, target)
```

对 Tiny 最优结构（比如 Tiny_R34_KD）至少跑一组 MSE vs MSE+SRCC 的对比实验。

---

### 六、可视化与日志（Tiny 专用）

确保 Tiny 系列训练过程中：

1. 使用 `tqdm` 显示进度条：
   - 每个 batch 显示：当前 loss、当前 lr、阶段（A/B）、mode。
2. 用 `logging_utils.plot_training_curves` 为每个 Tiny 模式画出：
   - train/val loss 曲线；
   - val_PLCC / val_SRCC 曲线；
3. 在 `eval_models.py` 中，对每个 Tiny 模型在每个数据集上：
   - 输出预测 vs MOS 散点图，文件命名例如：
     - `scatter_tiny_r18_kd_koniq.png`
     - `scatter_tiny_r18_kd_spaq.png` 等。

---

### 七、参数量与推理时间评估（Tiny vs Baseline）

在某个工具脚本里（例如 `src/utils/model_stats.py`）实现：

```python
def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference(model, input_size=(1, 3, 512, 512), device="cuda", n_warmup=10, n_iters=50):
    ...
    return avg_ms_per_image
```

并在一个 `scripts/collect_model_stats.py` 脚本中：

- 对 `resnet_baseline`, `tiny_r18`, `tiny_r18_kd`, `tiny_r34`, `tiny_r34_kd` 等模型：
  - 统计 trainable params；
  - 在统一 input_size 下测推理时间；
- 将统计结果写入一个 JSON，例如 `results/model_stats.json`。

---

### 八、实验与结果收集

你需要让项目支持一键或少量命令完成以下实验（可由 `run_ablation_suite.py` 调用）：

1. **KonIQ 上的性能比较**：
   - `resnet_baseline`
   - `tiny_r18`
   - `tiny_r18_kd_stageAB`
   - `tiny_r34`
   - `tiny_r34_kd_stageAB`
   - （可选）`tiny_r18_ms` / `tiny_r18_ms_kd`

2. **跨数据集泛化（KonIQ 训练 → SPAQ/KADID/AGIQA 测试）**：
   - 使用上述每个模型在 KonIQ 训练好的 ckpt；
   - 在 SPAQ / KADID / AGIQA 上直接评估。

3. **损失组合消融（以 Tiny 最优结构为例）**：
   - Tiny_best with MSE；
   - Tiny_best with MSE+SRCC。

所有结果汇总到一个最终 JSON，例如 `results/tiny_ablation_results.json`，结构类似：

```json
{
  "resnet_baseline": {
    "koniq": {"plcc": ..., "srcc": ..., "rmse": ..., "mae": ...},
    "spaq": {...},
    "kadid": {...},
    "agiqa": {...}
  },
  "tiny_r18": {...},
  "tiny_r18_kd": {...},
  "tiny_r34": {...},
  "tiny_r34_kd": {...}
}
```

再由 `make_tables.py` 转成 Markdown / LaTeX 表格，重点展示：

- 性能下降 vs 参数/时间节省；
- KD + 更强 Tiny 结构带来的改进。

---

### 九、风格与约束

- 不要删除已有功能，只做兼容性扩展；
- 所有新增文件和类要有清晰注释，说明与 TinyIQA / KD 的关系；
- 代码风格保持与现有项目一致（命名规则、import 风格等）；
- 所有路径配置要通过 YAML / 命令行管理，不要写死在代码里。

请根据以上要求修改并生成项目代码，使 TinyIQA 系列在保持轻量的前提下，通过改进的蒸馏与训练策略，获得尽可能接近 ResNet-50 baseline 的性能，并在 KonIQ / SPAQ / KADID / AGIQA 上得到系统性对比结果和可视化图像。
