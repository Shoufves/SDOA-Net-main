import time
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import math
import matplotlib.pyplot as plt


# 生成DOA并保持最小间隔
def gen_doa(target_num, doa, ant_num):
    doa_min = -45
    doa_max = 45
    # min_sep = 102.0 / ((ant_num-1)*super_ratio)
    min_sep = 102.0 / (ant_num - 1) * 2.0
    for n in range(target_num):
        condition = True
        doa_new = 0
        while condition:
            doa_new = np.random.rand() * (doa_max - doa_min) + doa_min
            condition = (np.min(np.abs(doa - doa_new)) < min_sep)
        doa[n] = doa_new
    # for n in range(target_num):
    #     doa_new = np.random.rand() * (doa_max - doa_min) + doa_min
    #     doa[n] = doa_new


# 理想阵列流形向量，对应公式2
def steer_vec(doa_deg, d, ant_num, d_per):
    st = np.exp(1j * 2 * np.pi * (d * np.arange(ant_num).T + d_per) * np.sin(np.deg2rad(doa_deg)))
    return st


# 核心信号生成器，生成一批次包含所有非理想因素的接收信号
def gen_signal(data_num, args):
    target_num = np.random.randint(1, args.max_target_num + 1, data_num)
    doa = np.ones((data_num, args.max_target_num)) * np.inf
    s = np.zeros((data_num, 2, args.ant_num))
    for n in range(data_num):
        gen_doa(target_num[n], doa[n], args.ant_num)
        # perturbation
        d_per = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_per_std
        # phase and amplitude
        amp = np.ones(args.ant_num).T + np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_amp_std
        pha = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_phase_std
        amp_phase = amp * np.exp(1j * pha)
        for m in range(target_num[n]):
            st = amp_phase * steer_vec(doa[n, m], args.d, args.ant_num, d_per)
            s[n, 0] = s[n, 0] + st.real
            s[n, 1] = s[n, 1] + st.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
        s_comp = s[n, 0] + 1j * s[n, 1]
        # mutual coupling
        max_mc_power = np.power(args.max_mc, np.arange(args.ant_num))
        mc_mat = np.zeros((args.ant_num, args.ant_num), dtype=complex)
        for idx_ant1 in range(args.ant_num):
            for idx_ant2 in range(args.ant_num):
                if idx_ant1 == idx_ant2:
                    mc_mat[idx_ant1, idx_ant2] = 1
                else:
                    mc_power = np.random.rand() * max_mc_power[np.abs(idx_ant2 - idx_ant1)]
                    mc_mat[idx_ant1, idx_ant2] = np.sqrt(mc_power) * np.exp(1j * np.random.rand(1) * 2 * np.pi)
        s_comp_mc = np.matmul(mc_mat, s_comp)
        s[n, 0] = s_comp_mc.real
        s[n, 1] = s_comp_mc.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

        # non linear function
        if args.is_nonlinear:
            s[n] = np.tanh(args.nonlinear*s[n])
            s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

    doa[doa == float('inf')] = -100
    doa = np.sort(doa, axis=1)

    # ---------- 1-bit 量化说明 ----------
    # 1-bit 量化已移至-加噪之后-执行（与测试流程一致，模拟真实ADC量化含噪接收信号）：
    #   - 训练：train_net 中 noise_torch 之后对 noisy_signal 取符号
    #   - 测试：main.py 中对 noisy_signals 取符号
    # gen_signal 只生成干净信号，不再在此量化。
    # ------------------------------------

    return s.astype('float32'), doa.astype('float32'), target_num


# 生成参考谱
def gen_refsp(doa, doa_grid, sigma):
    ref_sp = np.zeros((doa.shape[0], doa_grid.shape[0]))
    for i in range(doa.shape[1]):
        dist = np.abs(doa_grid[None, :] - doa[:, i][:, None])
        ref_sp += np.exp(- dist ** 2 / sigma ** 2)
    return ref_sp


# 添加高斯白噪声
def noise_torch(s, snr, snr_dB=None, snr_min_db=None, snr_max_db=30.0, snapshots=1):
    """加噪。三种模式：
       1) snr_dB 给定（测试）：σ = 10^(-snr_dB/20) 固定，只有噪声随机；
       2) snr_dB 为 None（训练）：SNR_dB ~ U(snr_min_db, snr_max_db) 逐样本采样；
       3) snr_dB 为 None 且 snr_min_db is None（旧行为兼容）：σ_max = sqrt(1/snr)，σ~U(0,σ_max)。
       snapshots>1（多快拍）：σ 每样本固定（同一物理场景），噪声逐快拍独立，
       返回 (B, L, 2, ant_num)；snapshots=1 时返回 (B, 2, ant_num) 兼容旧接口。
    """
    bsz, _, signal_dim = s.size()
    s_flat = s.view(bsz, -1)
    if snr_dB is not None:
        sigmas = torch.full((bsz,), 10.0 ** (-snr_dB / 20.0), device=s.device, dtype=s.dtype)
    elif snr_min_db is None:
        sigma_max = np.sqrt(1.0 / snr)
        sigmas = sigma_max * torch.rand(bsz, device=s.device, dtype=s.dtype)
    else:
        snr_db = snr_min_db + (snr_max_db - snr_min_db) * torch.rand(bsz, device=s.device, dtype=s.dtype)
        sigmas = torch.pow(10.0, -snr_db / 20.0)

    noise = torch.randn(bsz, snapshots, s_flat.shape[1], device=s.device, dtype=s.dtype)
    mult = sigmas[:, None, None] * torch.norm(s_flat, 2, dim=1)[:, None, None] / \
           (torch.norm(noise, 2, dim=2)[:, :, None] + 1e-8)
    noisy = (s_flat[:, None, :] + noise * mult).view(bsz, snapshots, -1, signal_dim)
    if snapshots == 1:
        return noisy[:, 0]  # (B, 2, ant_num) 兼容旧接口
    return noisy            # (B, L, 2, ant_num)


# Dither-CovNet 专用：多快拍 + 双独立均匀抖动 → 1-bit 协方差上三角
def dither_cov_feature(clean_signal, snapshots, T, snr_dB=None,
                       snr_min_db=0.0, snr_max_db=30.0):
    """clean_signal: (B, 2, ant_num)
       返回: (B, 272) 协方差特征（16 阵元复协方差上三角的实虚部）
    """
    bsz, _, ant_num = clean_signal.size()
    noisy = noise_torch(clean_signal, None, snr_dB=snr_dB,
                        snr_min_db=snr_min_db, snr_max_db=snr_max_db,
                        snapshots=snapshots)                  # (B, L, 2, N)

    tau1 = (torch.rand_like(noisy) * 2.0 - 1.0) * T
    tau2 = (torch.rand_like(noisy) * 2.0 - 1.0) * T

    q1 = torch.sign(noisy + tau1)                              # (B, L, 2, N)
    q2 = torch.sign(noisy + tau2)

    r1 = torch.complex(q1[:, :, 0, :], q1[:, :, 1, :])         # (B, L, N)
    r2 = torch.complex(q2[:, :, 0, :], q2[:, :, 1, :])

    R1 = T ** 2 * torch.einsum('bln,blm->bnm', r1.conj(), r2) / snapshots
    R = 0.5 * (R1 + R1.transpose(1, 2).conj())

    idx = torch.triu_indices(ant_num, ant_num, device=R.device)
    feat = torch.cat([R.real[:, idx[0], idx[1]],
                      R.imag[:, idx[0], idx[1]]], dim=1)
    return feat                                                # (B, 272)


# 轻量级Cross-Attention模块，用于捕获卷积层之间的长距离依赖关系
class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.d_model = 16
        self.num_heads = 4
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.in_proj = nn.Linear(in_dim, self.d_model, bias=False)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, in_dim, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        bsz, seq_len, _ = x.shape
        residual = x

        x = self.in_proj(x)  # (bsz, seq_len, d_model)即(bsz, 32, 16)

        # qkv shape: (bsz, 4, 32, 4) (批次, 4个头, 序列长度32, 每个头维度4)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k.transpose: (bsz, 4, 4, 32) 与q相乘变为(bsz, 4, 32, 32)
        # (32, 32)是注意力权重矩阵
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # 加权求和 (bsz, 4, 32, 32) @ (bsz, 4, 32, 4) = (bsz, 4, 32, 4)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model) # (bsz, 32, 16)
        out = self.out_proj(out)  # (bsz, seq_len, in_dim)即(bsz, 32, 2)

        return (out + residual).contiguous()  # 残差连接


# SDOA-Net的网络结构，输入为时域信号，输出为空间谱（含Cross-Attention）
class spectrumModule(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, in_dim=None):
        super().__init__()
        self.n_filters = n_filters
        self.in_layer = nn.Linear(in_dim if in_dim is not None else 2 * signal_dim,
                                  inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):  # padding=kernel_size - 1
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                          padding_mode='circular'),
                # nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        # 在卷积层之后、展平之前插入Cross-Attention，捕获阵元间长距离依赖
        self.attn = CrossAttention(n_filters)
        # self.out_layer1 = nn.ConvTranspose1d(n_filters, 1, 4, stride=1, padding=4 // 2, output_padding=1, bias=False)
        # self.linear1 = nn.Linear(inner_dim, 2 * signal_dim, bias=False)
        self.out_layer = nn.Linear(inner_dim * n_filters, 2 * signal_dim, bias=False)


    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)   # (bsz, 32)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)  # (bsz, 2, 32)
        x = self.mod(x)   # (bsz, 2, 32) 经过 6 层卷积
        # Cross-Attention: 将inner_dim视作序列长度，在n_filters特征维度上做多头注意力
        x = x.transpose(1, 2)          # (bsz, inner_dim, n_filters)即(bsz, 32, 2)
        x = self.attn(x)               # (bsz, inner_dim, n_filters)调用多头注意力，shape不变
        x = x.transpose(1, 2)          # (bsz, n_filters, inner_dim)即(bsz, 2, 32)
        x = x.reshape(bsz, -1)    # (bsz, 64) 展平
        x = self.out_layer(x).view(bsz, -1)     # (bsz, 32)
        return x


# 原版SDOA-Net（不含Cross-Attention），用于加载旧模型net0.pkl
class spectrumModuleV0(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):  # padding=kernel_size - 1
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                          padding_mode='circular'),
                # nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, 2 * signal_dim, bias=False)


    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = x.view(bsz, -1)
        x = self.out_layer(x).view(bsz, -1)
        return x


# 用于对比的深度频率方法
class DeepFreq(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    # padding = (kernel_out - upsampling + 1) // 2
    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x


# 从频谱估计DOA
def get_doa(sp, doa_num, doa_grid, max_target_num, ref_doa):
    est_doa = -100 * np.ones((doa_num.shape[0], max_target_num))
    for n in range(len(doa_num)):
        find_peaks_out = scipy.signal.find_peaks(sp[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(doa_num[n]))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        est_doa[n, :num_spikes] = np.sort(doa_grid[find_peaks_out[0][idx]])
        tmp = est_doa[n].copy()
        for idx_tmp in range(len(ref_doa[n])):
            est_doa[n, idx_tmp] = tmp[np.argmin(np.abs(ref_doa[n, idx_tmp] - tmp))]

    # est_doa = np.sort(est_doa, axis=1)
    return est_doa


# 一个epoch的训练&验证流程
def train_net(args, net, optimizer, criterion, train_loader, val_loader,
              doa_grid, epoch, train_num, train_type, net_type):
    epoch_start_time = time.time()
    net.train()
    loss_train = 0
    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    if net_type == 0:
        for n in range(doa_grid.size):
            tmp = steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
            dic_mat[n, 0] = tmp.real
            dic_mat[n, 1] = tmp.imag
        dic_mat_torch = torch.from_numpy(dic_mat).float()
        if args.use_cuda:
            dic_mat_torch = dic_mat_torch.cuda()

    for batch_idx, (clean_signal, target_sp, doa) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_sp = clean_signal.cuda(), target_sp.cuda()
        if getattr(args, 'input_mode', 'signal') == 'cov':
            x_in = dither_cov_feature(clean_signal, args.snapshots, args.dither_T,
                                      snr_dB=None,
                                      snr_min_db=getattr(args, 'snr_min_db', 0.0),
                                      snr_max_db=getattr(args, 'snr_max_db', 30.0))
        else:
            noisy_signal = noise_torch(clean_signal, args.snr,
                                       snr_dB=None,
                                       snr_min_db=getattr(args, 'snr_min_db', None),
                                       snr_max_db=getattr(args, 'snr_max_db', 30.0))
            # 1-bit 量化在加噪之后执行（与测试一致：ADC量化含噪接收信号）
            if args.is_1bit:
                noisy_signal = torch.sign(noisy_signal)
            x_in = noisy_signal
        optimizer.zero_grad()
        output_net = net(x_in).view(args.batch_size, 2, -1)

        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
            # loss = criterion(torch.pow(mm_real, 2) + torch.pow(mm_imag, 2), target_sp)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)

        loss.backward()
        optimizer.step()
        loss_train += loss.data.item()

        # plt.figure()
        # plt.plot(sp.cpu().detach().numpy()[0])
        # plt.plot(target_sp.cpu().detach().numpy()[0])
        # plt.show()

    net.eval()
    loss_val, fnr_val = 0, 0
    for batch_idx, val_batch in enumerate(val_loader):
        if getattr(args, 'input_mode', 'signal') == 'cov':
            clean_signal, target_sp, doa = val_batch
            if args.use_cuda:
                clean_signal, target_sp = clean_signal.cuda(), target_sp.cuda()
            x_in = dither_cov_feature(clean_signal, args.snapshots, args.dither_T,
                                      snr_dB=None,
                                      snr_min_db=getattr(args, 'snr_min_db', 0.0),
                                      snr_max_db=getattr(args, 'snr_max_db', 30.0))
            if args.use_cuda:
                x_in = x_in.cuda()
        else:
            if len(val_batch) == 4:
                noisy_signal, _, target_sp, doa = val_batch
                if args.use_cuda:
                    noisy_signal, target_sp = noisy_signal.cuda(), target_sp.cuda()
            else:
                clean_signal, target_sp, doa = val_batch
                if args.use_cuda:
                    clean_signal, target_sp = clean_signal.cuda(), target_sp.cuda()
                noisy_signal = noise_torch(clean_signal, args.snr,
                                           snr_dB=None,
                                           snr_min_db=getattr(args, 'snr_min_db', None),
                                           snr_max_db=getattr(args, 'snr_max_db', 30.0))
            # 1-bit 量化在加噪之后执行（与训练/测试一致）
            if args.is_1bit:
                noisy_signal = torch.sign(noisy_signal)
            x_in = noisy_signal
        with torch.no_grad():
            output_net = net(x_in).view(args.batch_size, 2, -1)

        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)
        loss_val += loss.data.item()

        doa_num = (doa >= -90).sum(dim=1)
        est_doa = get_doa(sp.cpu().detach().numpy(), doa_num, doa_grid, args.max_target_num, doa)

    loss_train /= args.n_training
    loss_val /= args.n_validation

    print("TTrain_Num: %d, rain_Type: %d, Epochs: %d / %d, Time: %.1f, Training Loss: %.2f, Validation Loss:  %.2f" % (
        train_num,
        train_type,
        epoch, args.n_epochs,
        time.time() - epoch_start_time,
        loss_train,
        loss_val))
    # print(np.sort(doa[0]))
    # print(np.sort(est_doa[0]))
    return net, loss_train, loss_val
