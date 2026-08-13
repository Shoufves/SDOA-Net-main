# SDOA-Net: 面向非理想阵列的高效深度学习 DOA 估计网络（含 Cross-Attention 与 1-bit 量化扩展）

## 摘要

波达方向（DOA）估计在传统雷达、无线通信以及感知与通信一体化（ISAC）系统中都是至关重要的问题。然而，低成本系统往往存在各种非理想因素，例如天线位置扰动、互耦效应、增益/相位不一致以及非线性放大器效应，这些因素会显著降低 DOA 估计的性能。本文提出一种基于深度学习的超分辨 DOA 估计网络——SDOA-Net，用于更精确地表征真实阵列。与现有基于深度学习的 DOA 方法不同，SDOA-Net 直接使用采样接收信号（而非协方差矩阵）作为输入来提取数据特征。此外，SDOA-Net 输出一个与目标 DOA 无关的向量，该向量可用于估计空间谱，因此**同一个训练好的网络可以适用于任意数量的目标**，降低了实现的复杂度。所提出的 SDOA-Net 网络结构低维、收敛速度快于现有深度学习方法。仿真结果表明，在非理想阵列条件下，SDOA-Net 优于现有 DOA 估计方法。

## 项目结构

```
SDOA-Net-Attn/
├── doasys.py          # 信号生成、网络结构（spectrumModule + CrossAttention）、训练/验证流程
├── train.py           # 训练入口（支持 1-bit 量化开关、条件保存模型）
├── main.py            # 测试入口（浮点/1-bit 双环境对比、输出两张 RMSE 图）
├── requirements.txt   # conda 依赖
├── README.md          # 本文件
├── net.pkl            # 浮点训练模型
├── net_1bit.pkl       # 1-bit 训练模型（--is_1bit 1 时保存）
├── net0.pkl           # 原版 SDOA-Net（不含 Cross-Attention，可选对比）
└── pretrained/        # 预训练模型目录
```

## 网络结构

### spectrumModule（含 Cross-Attention 的改进版）

```
输入: (batch, 2, ant_num=16)          → flatten: (batch, 32)
      ↓
in_layer: Linear(32, 64)              → reshape: (batch, n_filters=2, inner_dim=32)
      ↓
6× [Conv1d(2→2, k=3, padding='circular') + BatchNorm + ReLU]
      ↓                                  ← 形状保持 (batch, 2, 32)
transpose: (batch, 32, 2)
      ↓
CrossAttention(in_dim=2)               ← 在 inner_dim=32 维度上做多头自注意力
      |  d_model=16, num_heads=4, head_dim=4
      |  捕获所有 32 个位置之间的长距离依赖（本版本与原版 SDOA-Net 的唯一结构差异）
      ↓
transpose: (batch, 2, 32)              → flatten: (batch, 64)
      ↓
out_layer: Linear(64, 32)              → reshape: (batch, 32) = (batch, 2×ant_num)
      ↓
输出: (batch, 2, ant_num=16)           ← 空间谱表示向量 h
```

**关键设计**：网络输出向量 `h` 与理想阵列流形字典矩阵 `dic_mat` 相乘得到空间谱 `sp = |mm_real|² + |mm_imag|²`，再通过 `find_peaks` 提取 DOA。`inner_dim = 32 = 2 × ant_num`，Conv1d 沿此序列方向卷积，Cross-Attention 捕获长距离依赖。

### 原版 spectrumModuleV0

不含 Cross-Attention，用于加载旧模型 `net0.pkl` 进行对比。

## 1-bit 量化扩展（本次新增）

### 动机

模拟极端低成本 ADC：将接收信号的实部/虚部分别取符号，量化为 ±1，作为新的非理想阵列因素引入 SDOA-Net 训练框架，验证改进网络对极端量化噪声的鲁棒性。

### 数据流程（训练/测试一致）

```
干净信号 → 加噪声（noise_torch）→ 【1-bit 量化 sign】→ 网络
```

⚠️ **1-bit 量化必须在加噪之后执行**（与真实 ADC 量化含噪接收信号的物理过程一致）。早期实现错误地在 `gen_signal` 中对干净信号量化，导致训练/测试分布不匹配、模型对比曲线重合，已修复。

### 训练命令

```bash
# 基础模型（从零训练 3 轮）
python train.py --new_train 1 --train_num 3 --n_epochs 100 --is_1bit 0

# 1-bit 模型（基础模型 + 10 轮 1-bit 训练，保存为 net_1bit.pkl）
python train.py --new_train 0 --train_num 10 --n_epochs 100 --is_1bit 1

# 浮点模型（基础模型 + 10 轮浮点训练，保存为 net.pkl）
python train.py --new_train 0 --train_num 10 --n_epochs 100 --is_1bit 0
```

### 测试命令

```bash
python main.py --is_1bit_test 1
```

当前实现输出两张图（供过程分析）：
- `RMSE_comparison_float.png`：浮点信号环境（SDOA+Attn / FFT / OMP 均处理浮点信号）
- `RMSE_comparison_1bit.png`：1-bit 信号环境（1-bit 训练模型 / 浮点模型退化基线 / FFT / OMP 均处理 1-bit 信号）

> **最终汇报图（老师要求）**：只需画**一张** RMSE 图，包含**六条曲线**：
> 1. 单比特信号环境下的 FFT（FFT + 1-bit 信号）
> 2. 单比特信号环境下的 OMP（OMP + 1-bit 信号）
> 3. 原始信号环境下的 FFT（FFT + 浮点信号）
> 4. 原始信号环境下的 OMP（OMP + 浮点信号）
> 5. 原始信号环境下训练与测试的 SDOA+Attn（浮点模型 + 浮点信号）
> 6. 单比特信号环境下训练与测试的 SDOA+Attn-1bit（1-bit 模型 + 1-bit 信号）
>
> 这样可在同一张图中同时对比「信号环境（浮点 vs 1-bit）」与「方法（深度学习 vs 传统）」两个维度。

## 实验结果（本轮）

| 对比项 | RMSE（典型值） | 说明 |
|---|---|---|
| 浮点模型 + 浮点信号 | ~0.6° | 上限基准 |
| 1-bit 训练模型 + 1-bit 信号 | ~2.9° | 性能损失明显 |
| FFT/OMP + 1-bit 信号 | 显著更差 | 深度学习优于传统方法 |

- 1-bit 训练模型优于传统方法（FFT/OMP）在 1-bit 信号上的表现，验证了深度学习对极端量化噪声的鲁棒性；
- 但 1-bit 量化带来 ~2.3° 的精度损失，在 DOA 估计中尚不可接受，需要进一步研究。

## 本次 1-bit 实验遇到的问题与注意事项（重要）

### 已解决的问题

1. **训练/测试量化位置不一致（Bug）**
   - 现象：浮点模型与 1-bit 模型在 1-bit 测试下的 RMSE 曲线基本重合，无法体现差异；
   - 原因：训练时在 `gen_signal` 中量化干净信号，测试时在加噪后量化含噪信号，两者分布不匹配；
   - 修复：统一在加噪之后量化（`doasys.py` 的 `train_net` 训练/验证循环）。

2. **对比不公平**
   - 现象：SDOA+Attn (1-bit) 的 RMSE 高于 FFT，看似"1-bit 引入无意义"；
   - 原因：FFT/OMP 误用了浮点信号，而网络用 1-bit 信号；
   - 修复：FFT/OMP 在 1-bit 图中改用 1-bit 信号，保证同环境对比。

3. **模型文件覆盖**
   - 现象：浮点模型与 1-bit 模型互相覆盖；
   - 修复：`train.py` 按 `--is_1bit` 条件保存 `net.pkl` / `net_1bit.pkl`。

### 注意事项（供后续实验参考）

1. **量化位置一致性**：训练、验证、测试三者的 1-bit 量化必须都在加噪之后执行，否则分布不匹配；
2. **训练轮次充足性**：从零训练（`--new_train 1`）需要足够轮数，欠拟合会导致 RMSE 曲线平直、不随 SNR 下降；
3. **对比环境一致性**：网络与传统方法必须使用同一种信号（同为浮点或同为 1-bit），否则对比不公平；
4. **训练噪声参数 `--snr`**：默认 `snr=1.0` 训练噪声上限过宽（sigma_max=1.0），测试 SNR（0~30dB）全部落在训练分布低噪声端，导致曲线平直；可适当提高 `--snr` 或使用预训练模型微调；
5. **迁移学习策略**：建议先 `--new_train 1 --train_num 3` 训练基础模型，再分别用 `--new_train 0 --train_num 10 --is_1bit 1/0` 训练 1-bit 与浮点模型，保证同起点、同轮次、公平对比；
6. **1-bit 信息论极限**：1-bit 量化后每根天线仅 4 种状态（±1±j），可编码的角度信息存在硬上限，单纯增加训练难以完全恢复浮点精度。

### 后续研究方向

1. **信号预处理**：先对 1-bit 接收信号做预处理（如平滑、滤波、重构）再送入网络训练；
2. **渐进量化训练**：先用高精度（8-4-2 bit）量化训练若干轮，再用 1-bit 量化做主要训练（迁移学习），期望模型学到更多规律；
3. **查阅 1-bit DOA 估计相关论文**，借鉴提高单比特 DOA 精度的办法；
4. **评价标准**：若 1-bit 曲线与浮点曲线 RMSE 相差不大，即可视为很大成果（1-bit ADC 成本极低而性能接近），应以此为目标。

## 环境依赖

```bash
# GPU 版本（推荐）
conda create -n sdoa python=3.10
conda install pytorch>=2.0 cudatoolkit=12.1 -c pytorch -c nvidia
conda install --file requirements.txt

# CPU 版本
conda install --file requirements.txt -c pytorch
```

> 注：Windows 环境下若出现 `libomp.dll` 与 `libiomp5md.dll` 的 OpenMP 冲突告警，可设置 `set KMP_DUPLICATE_LIB_OK=TRUE` 或安装 `nomkl` 缓解；该冲突不影响已确认的数值结果。
