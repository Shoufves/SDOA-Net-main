# Dither-CovNet 实验方案

> 面向 1-bit ADC 的 SDOA 高精度 DOA 估计  
> 目的：在有限实验时间内给出一个能显著压低 RMSE、且高 SNR 不恶化的单一实验方案。

---

## 1. 实验动机

### 1.1 当前问题

- 现有 1-bit SDOA 路线在 P0/P1/P2 之后 RMSE 仍在 **1.2° 以上**；
- 高 SNR 段 RMSE 反而随 SNR 上升；
- 根因已定位为 **mean-sign 预处理在高 SNR 下失效**：

```text
高 SNR 时 noise→0
sign(s + noise) ≈ sign(s)
mean_L(sign(s+noise)) ≈ sign(s)
```

即幅度信息完全丢失，网络只能看到硬符号。

### 1.2 老师的核心要求

1. 单比特信号 DOA 估计误差需要降到 **零点几甚至零点零几**；
2. 若网络能把“非线性单比特输入”映射回“线性数学空间”，则不必依赖环境噪声制造符号翻转。

### 1.3 本实验的解决思路

不再让网络直接吃 1-bit 符号或 mean-sign，而是先用 **已知均匀抖动 + 双路量化** 把 1-bit 测量转换为 **线性协方差矩阵**，再让网络从协方差学 DOA。

---

## 2. 主要方法：Dither-CovNet

### 2.1 数据生成

每个样本包含：

- 同一组 DOA、阵列非理想参数（位置扰动、增益/相位、互耦、tanh 非线性）；
- L 个快拍，快拍间仅噪声/抖动不同；
- 每个快拍生成两个独立均匀抖动 τ1, τ2；
- 分别量化得到 r1, r2。

### 2.2 协方差估计

采用文献中基于均匀抖动的无偏协方差估计：

```text
R̂1 = T²/L · Σ_l r1(l) r2^H(l)
R̂  = 0.5 · (R̂1 + R̂1^H)
```

其中 T 为均匀抖动幅度，r1/r2 为 1-bit 量化后的复信号：

```text
r = sign(Re{x}) + j · sign(Im{x})
```

将 `R̂` 的复协方差上三角作为网络输入：

```text
特征维度 = 2 × N(N+1)/2 = 272   （N=16 阵元）
```

### 2.3 为什么有效

- 抖动本身是已知的、可控的随机源，**不再依赖环境噪声制造翻转**；
- 即使高 SNR 下环境噪声很小，抖动仍能让 1-bit 输出携带统计信息；
- 协方差域是**线性数学空间**，网络学起来比直接学非线性量化映射容易得多；
- 多快拍 L 越大，协方差估计越准，因此 RMSE 应随 L 单调下降。

---

## 3. 当前网络原理

### 3.1 主干

沿用 SDOA-Net + Cross-Attention 主干：

```text
输入: 协方差上三角 (B, 272)
  ↓
in_layer: Linear(272 → 64)
  ↓ reshape (B, 2, 32)
  ↓ 6×[Conv1d(2→2, k=3, circular) + BN + ReLU]
  ↓ transpose (B, 32, 2)
  ↓ CrossAttention(d_model=16, 4 heads)
  ↓ transpose + flatten (B, 64)
  ↓ out_layer: Linear(64 → 32)
  ↓ h (B, 32)
```

### 3.2 输出

保留原 SDOA-Net 的空间谱构造方式：

```text
sp(θ) = |a^H(θ) h|²
```

然后通过 `find_peaks` 提取 DOA。  
因此同一个训练好的网络仍然支持 1~3 个目标。

### 3.3 与旧网络的关系

- 从零开始训练，不加载旧 `net.pkl`；
- 仅把网络第一层 `in_layer` 的输入从 32 维改为 272 维；
- 其余卷积层、注意力层、输出层保持 SDOA-Net + Cross-Attention 原结构。

> 网络原理图：`Dither-CovNet_网络原理图.tex`，可用 `xelatex Dither-CovNet_网络原理图.tex` 编译。

---

## 4. 实验设置

### 4.1 训练

**建议使用 `--new_train 1` 从零开始训练，并采用 P0 风格的训练配置：SNR 均匀采样 + 7 步非理想渐进训练。**

```bash
# 主实验：从零训练，L=256，100 epochs，10 轮 7 步渐进训练
# 实际训练量 = 10 × 7 × 100 = 7000 epoch
python train.py \
  --new_train 1 \
  --input_mode cov \
  --snapshots 256 \
  --dither_T 3.0 \
  --snr_min_db 0 \
  --snr_max_db 30 \
  --single_stage -1 \
  --train_num 10 \
  --n_epochs 100 \
  --tag CovL256
```

如果时间不够，可以先跑少量轮次验证流程：

```bash
# 快速验证：只跑 3 轮，2100 epoch
python train.py \
  --new_train 1 \
  --input_mode cov \
  --snapshots 256 \
  --dither_T 3.0 \
  --snr_min_db 0 \
  --snr_max_db 30 \
  --single_stage -1 \
  --train_num 3 \
  --n_epochs 100 \
  --tag CovL256x3
```

如果时间极紧，可用单阶段快速验证：

```bash
# 只跑“全非理想”阶段，30 epoch
python train.py \
  --new_train 1 \
  --input_mode cov \
  --snapshots 256 \
  --dither_T 3.0 \
  --snr_min_db 0 \
  --snr_max_db 30 \
  --single_stage 6 \
  --train_num 1 \
  --n_epochs 30 \
  --tag CovL256_quick
```

#### 超参数含义

| 参数 | 含义 |
|---|---|
| `--new_train 1` | 从零创建网络，不加载旧 `net.pkl` |
| `--input_mode cov` | 使用 Dither-CovNet：输入为抖动协方差上三角 |
| `--snapshots 256` | 协方差估计使用的快拍数 L；L 越大，估计越准 |
| `--dither_T 3.0` | 均匀抖动幅度 T；T=3 能覆盖归一化信号的动态范围，保证协方差估计近似无偏 |
| `--snr_min_db 0` | 训练噪声 SNR 下限 0 dB |
| `--snr_max_db 30` | 训练噪声 SNR 上限 30 dB |
| `--single_stage -1` | 走完整 7 步非理想渐进训练；若想快速验证可改为 `6` 只跑全非理想一步 |
| `--train_num 10` | 完整 7 步渐进训练的重复次数；即每个非理想阶段重复训练轮数 |
| `--n_epochs 100` | 每个训练阶段内的 epoch 数 |
| `--tag` | 模型文件名后缀，避免覆盖 |

> **实际训练 epoch 数计算：**  
> 当 `--single_stage -1` 时共有 7 个非理想阶段。  
> 主实验 `--train_num 10 --n_epochs 100` 对应的实际训练量为：  
> **10 轮 × 7 阶段 × 100 epoch = 7000 epoch**。  
> `--train_num 3 --n_epochs 100` 对应：  
> **3 轮 × 7 阶段 × 100 epoch = 2100 epoch**。  
> 因此 `--train_num` 控制的是“完整 7 步渐进训练重复几遍”，而不是只训练 1 个 epoch。

> 说明：Dither-CovNet 的“P0 化训练”体现在训练集 SNR 按 0~30 dB 均匀采样（与测试一致），以及保留原 SDOA-Net 的 7 步非理想渐进训练；因为协方差输入已经完成 1-bit 非线性线性化，所以不再需要额外做量化位宽课程。

### 4.2 测试

```bash
python main.py \
  --input_mode cov \
  --model_cov net_1bit_cov_L256_T3p0CovL256.pkl \
  --cov_L_list 16,64,256 \
  --dither_T 3.0
```

该命令会输出两张图：

1. `RMSE_Dither_CovNet_comparison.png`  
   最大 L 的 Dither-CovNet vs FFT/OMP 1-bit。

2. `RMSE_Dither_CovNet_Lsweep.png`  
   同一个 Dither-CovNet 网络在不同 L 下的 RMSE 曲线，体现多快拍收益。

---

## 5. 实验产出

### 5.1 曲线图

| 文件名 | 内容 |
|---|---|
| `RMSE_Dither_CovNet_comparison.png` | Dither-CovNet（最大 L）与 FFT、OMP 在 1-bit 环境下的 RMSE-SNR 对比 |
| `RMSE_Dither_CovNet_Lsweep.png` | 同一网络在不同快拍数 L 下的 RMSE-SNR 对比 |

### 5.2 数据表

建议在汇报中输出：

| L | 0 dB | 10 dB | 20 dB | 30 dB |
|---|---:|---:|---:|---:|
| 16 | | | | |
| 64 | | | | |
| 256 | | | | |
| FFT 1-bit | | | | |
| OMP 1-bit | | | | |

### 5.3 预期结论

- 高 SNR 不应再出现 RMSE 上升；
- RMSE 随 L 增大而下降；
- 最大 L 的 Dither-CovNet 应显著优于 FFT/OMP 1-bit；
- 当 L 达到 64 以上时，有机会进入 **零点几度** 区间。

---

## 6. 注意事项

1. 训练/测试必须使用相同的 `--dither_T` 和 `--input_mode cov`；
2. 模型命名中 L 表示训练时使用的快拍数；测试时的 L 扫描也可用同一网络，因为协方差特征维度固定；
3. 如果运行时间不足，可先用 `--cov_test_rounds 2`、`--cov_test_len 1000` 快速出图。
