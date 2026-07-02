import os
import sys
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch.utils.data as data_utils
import torch.nn as nn
import scipy.signal
import math
import doasys
#import doa_method 没传这个文件
from scipy import io

import matlab.engine  # 服务器无MATLAB许可证，已关闭


def make_hankel(signal, m):
    """
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype='complex128')
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(-2.0j * np.pi * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1)))
        u = V[nfreq[n]:]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr


# 加载旧版SDOA-Net模型的辅助函数（net0.pkl由原版doasys.spectrumModule保存）
def load_legacy_model(path, use_cuda):
    _orig_class = doasys.spectrumModule
    doasys.spectrumModule = doasys.spectrumModuleV0
    if use_cuda:
        net = torch.load(path, weights_only=False)
    else:
        net = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    doasys.spectrumModule = _orig_class
    return net


if __name__ == '__main__':

    is_music = False  # 服务器无MATLAB许可证，已关闭MUSIC对比
    is_anm = False    # 服务器无MATLAB许可证，已关闭ANM对比
    is_proposed = True
    is_legacy = True  # 是否加载原版SDOA-Net（net0.pkl）进行对比

    is_fig = True
    is_save = False

    parser = argparse.ArgumentParser()

    # parser.add_argument('--numpy_seed', type=int, default=12345)  # 222
    # parser.add_argument('--torch_seed', type=int, default=12345)  # 333

    parser.add_argument('--n_training', type=int, default=8000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=640, help='# of validation data')



    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=int, default=40, help='the size of grids')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')

    # module parameters
    parser.add_argument('--n_layers', type=int, default=8, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=8, help='number of filters per layer in the module')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs used to train the module')

    # array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    # parser.add_argument('--super_ratio', type=float, default=1, help='super-resolution ratio based on 102/(ant_num-1)')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--snr', type=float, default=1., help='the maximum SNR')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # imperfect parameters 0.15 0.5 0.2
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    # training policy
    parser.add_argument('--new_train', type=int, default=0, help='train a new network')
    parser.add_argument('--net_type', type=int, default=0, help='the type of network')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    # np.random.seed(args.numpy_seed)
    # torch.manual_seed(args.torch_seed)

    doa_grid = np.linspace(-50, 50, args.grid_size, endpoint=False)
    # ref_grid = np.linspace(-50, 50, 16, endpoint=False)
    ref_grid = doa_grid
    # generate the training data

    # 加载新版SDOA-Net（含Cross-Attention）
    if args.use_cuda:
        net = torch.load('net.pkl', weights_only=False)
    else:
        net = torch.load('net.pkl', map_location=torch.device('cpu'), weights_only=False)

    # 加载原版SDOA-Net（不含Cross-Attention，用于对比）
    if is_legacy:
        net0 = load_legacy_model('net0.pkl', args.use_cuda)
        print("Loaded original SDOA-Net from net0.pkl")

    # PyTorch 2.6+ compatibility
    for module in net.modules():
        if isinstance(module, nn.Conv1d) and not hasattr(module, '_reversed_padding_repeated_twice'):
            p = module.padding
            module._reversed_padding_repeated_twice = (p, p) if isinstance(p, int) else tuple(p) * 2
    if is_legacy:
        for module in net0.modules():
            if isinstance(module, nn.Conv1d) and not hasattr(module, '_reversed_padding_repeated_twice'):
                p = module.padding
                module._reversed_padding_repeated_twice = (p, p) if isinstance(p, int) else tuple(p) * 2

    if args.use_cuda:
        net.cuda()
        if is_legacy:
            net0.cuda()

    # 多GPU支持：双卡3090 DataParallel
    if args.use_cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUs: %s" % (torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        net = nn.DataParallel(net)
        if is_legacy:
            net0 = nn.DataParallel(net0)

    # 构建字典矩阵，公式26
    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    dic_mat_comp = np.zeros((doa_grid.size, args.ant_num), dtype=complex)
    for n in range(doa_grid.size):
        tmp = doasys.steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
        tmp = tmp / np.sqrt(np.sum(np.power(np.abs(tmp), 2)))
        dic_mat[n, 0] = tmp.real
        dic_mat[n, 1] = tmp.imag
        dic_mat_comp[n] = tmp
    dic_mat_torch = torch.from_numpy(dic_mat).float()
    if args.use_cuda:
        dic_mat_torch = dic_mat_torch.cuda()

    # generate the validation data
    # SNR_range = np.linspace(0, 30, 4)
    SNR_range = np.linspace(10, 30, 7)
    RMSE = np.zeros((SNR_range.size, 1))
    RMSE_legacy = np.zeros((SNR_range.size, 1))
    RMSE_FFT = np.zeros((SNR_range.size, 1))
    RMSE_MUSIC = np.zeros((SNR_range.size, 1))
    RMSE_OMP = np.zeros((SNR_range.size, 1))
    RMSE_ANM = np.zeros((SNR_range.size, 1))

    eng = matlab.engine.start_matlab()           # 服务器无MATLAB许可证，已关闭
    eng.cvx_startup(nargout=0)                   # 服务器无MATLAB许可证，已关闭

    # dic_music = np.zeros((doa_grid.size, antnum_reshape), dtype=complex)
    # for idx2 in range(doa_grid.size):
    #     dic_music[idx2] = doasys.steer_vec(doa_grid[idx2], args.d, antnum_reshape, np.zeros(antnum_reshape).T)

    for n in range(SNR_range.size):
        n_test = 100 # 10
        RMSE[n] = 0
        RMSE_legacy[n] = 0
        RMSE_FFT[n] = 0
        RMSE_MUSIC[n] = 0
        RMSE_OMP[n] = 0
        RMSE_ANM[n] = 0
        for n1 in range(n_test):
            epoch_start_time = time.time()
            test_len = 2000
            signal, doa, target_num = doasys.gen_signal(test_len, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            # 添加噪声
            SNR_dB = SNR_range[n]
            noisy_signals = doasys.noise_torch(signal, math.pow(10.0, SNR_dB / 10.0))

            if is_proposed:
                if args.use_cuda:
                    noisy_signals = noisy_signals.cuda()
                with torch.no_grad():
                    output_net = net(noisy_signals).view(test_len, 2, -1)

                mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 1, :].T)
                mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 0, :].T)
                sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
                sp_np = sp.cpu().detach().numpy()
                for idx_sp in range(sp_np.shape[0]):
                    sp_np[idx_sp] = sp_np[idx_sp] / np.max(sp_np[idx_sp])

                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_np, doa_num, doa_grid, args.max_target_num, doa)
                RMSE[n] = RMSE[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # 原版SDOA-Net（不含Cross-Attention）对比
            if is_legacy:
                if args.use_cuda:
                    noisy_signals_legacy = noisy_signals
                else:
                    noisy_signals_legacy = noisy_signals
                with torch.no_grad():
                    output_net0 = net0(noisy_signals_legacy).view(test_len, 2, -1)

                mm_real0 = torch.mm(output_net0[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net0[:, 1, :],
                                                                                               dic_mat_torch[:, 1, :].T)
                mm_imag0 = torch.mm(output_net0[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net0[:, 1, :],
                                                                                               dic_mat_torch[:, 0, :].T)
                sp0 = torch.pow(mm_real0, 2) + torch.pow(mm_imag0, 2)
                sp0_np = sp0.cpu().detach().numpy()
                for idx_sp in range(sp0_np.shape[0]):
                    sp0_np[idx_sp] = sp0_np[idx_sp] / np.max(sp0_np[idx_sp])

                est_doa0 = doasys.get_doa(sp0_np, doa_num, doa_grid, args.max_target_num, doa)
                RMSE_legacy[n] = RMSE_legacy[n] + np.sum(np.power(np.abs(est_doa0 - doa), 2))

            # FFT method
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            sp_FFT = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(r_c).T)), 2).T

            for idx_sp in range(sp_FFT.shape[0]):
                sp_FFT[idx_sp] = sp_FFT[idx_sp] / np.max(sp_FFT[idx_sp])
            doa_num = (doa >= -90).sum(axis=1)
            est_doa = doasys.get_doa(sp_FFT, doa_num, doa_grid, args.max_target_num, doa)
            RMSE_FFT[n] = RMSE_FFT[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # MUSIC alg  -- 服务器无MATLAB许可证，已关闭
            if is_music:
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()

                r_c = r[:, 0, :] + 1j * r[:, 1, :]
                sp_MUSIC = np.zeros((r_c.shape[0], args.grid_size))
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.MUSIConesnapshot(matlab.double(list(r_c[idx_r]), is_complex=True),
                                                 int(target_num[idx_r]),
                                                 matlab.double(list(doa_grid), is_complex=False))
                    sp_MUSIC[idx_r] = np.squeeze(np.asarray(x_tmp))

                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_MUSIC, doa_num, doa_grid, args.max_target_num, doa)
                RMSE_MUSIC[n] = RMSE_MUSIC[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # OMP alg
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            est_doa_omp = -100 * np.ones((r_c.shape[0], args.max_target_num))
            for idx1 in range(r_c.shape[0]):
                r_tmp0 = np.expand_dims(r_c[idx1], axis=0)
                r_tmp1 = r_tmp0
                max_idx = np.zeros(target_num[idx1], dtype=int)
                for idx2 in range(target_num[idx1]):
                    max_idx_tmp = np.argmax(np.abs(np.matmul(dic_mat_comp, np.conj(r_tmp1).T)))
                    max_idx[idx2] = max_idx_tmp
                    dic_tmp = dic_mat_comp[max_idx[0:idx2 + 1]]
                    r_tmp1 = r_tmp0 - np.matmul(np.matmul(r_tmp0, np.linalg.pinv(dic_tmp)), dic_tmp)
                    est_doa_omp[idx1, idx2] = doa_grid[max_idx_tmp]
                est_doa_omp[idx1] = np.sort(est_doa_omp[idx1])
            RMSE_OMP[n] = RMSE_OMP[n] + np.sum(np.power(np.abs(est_doa_omp - doa), 2))

            # atomic norm minimization alg  -- 服务器无MATLAB许可证，已关闭
            if is_anm:
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()
                r_c = r[:, 0, :] + 1j * r[:, 1, :]
                x = np.zeros((r_c.shape[0], args.ant_num), dtype=complex)
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.ANM(matlab.double(list(r_c[idx_r]), is_complex=True))
                    x[idx_r] = np.squeeze(np.asarray(x_tmp))

                sp_ANM = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(x).T)), 2).T

                for idx_sp in range(sp_ANM.shape[0]):
                    sp_ANM[idx_sp] = sp_ANM[idx_sp] / np.max(sp_ANM[idx_sp])
                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_ANM, doa_num, doa_grid, args.max_target_num, doa)
                RMSE_ANM[n] = RMSE_ANM[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            if is_fig:
                plt.figure()
                if is_proposed:
                    plt.plot(doa_grid, sp_np[0], label='SDOA-Net + Cross-Attn')
                if is_legacy:
                    plt.plot(doa_grid, sp0_np[0], label='SDOA-Net (original)')
                plt.plot(doa_grid, sp_FFT[0], label='FFT method')
                if is_anm:
                    plt.plot(doa_grid, sp_ANM[0], label='ANM method')
                if is_music:
                    plt.plot(doa_grid, sp_MUSIC[0], label='MUSIC method')
                tmp_doa = est_doa_omp[0][np.argwhere(est_doa_omp[0] > -90)]
                if tmp_doa.size==3:
                    io.savemat('sp_OMP.mat', {'array': tmp_doa})
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), label='OMP method')

                tmp_doa = doa[0][np.argwhere(doa[0] > -90)]
                if tmp_doa.size == 3:
                    io.savemat('truth.mat', {'array': tmp_doa})
                plt.stem(tmp_doa, np.ones((tmp_doa.size, 1)), label='Ground-truth DOA')
                plt.xlabel('Spatial angle (deg)')
                plt.ylabel('Spatial spectrum')
                plt.legend()
                plt.grid()
                #plt.show() 不让他全部显示了，只要最后结果
                if tmp_doa.size == 3:
                    io.savemat('doa_grid.mat', {'array': doa_grid})
                    io.savemat('sp_proposed.mat', {'array': sp_np[0]})
                    if is_legacy:
                        io.savemat('sp_legacy.mat', {'array': sp0_np[0]})
                    if is_anm:
                        io.savemat('sp_ANM.mat', {'array': sp_ANM[0]})
                    if is_music:
                        io.savemat('sp_MUSIC.mat', {'array': sp_MUSIC[0]})
                    io.savemat('sp_FFT.mat', {'array': sp_FFT[0]})

            print("SNR: %.2f dB, Test: %d/%d, Time: %.2f" % (SNR_dB, n1, n_test, time.time() - epoch_start_time))
        RMSE[n] = np.sqrt(RMSE[n] / (doa.size * n_test))
        RMSE_legacy[n] = np.sqrt(RMSE_legacy[n] / (doa.size * n_test))
        RMSE_FFT[n] = np.sqrt(RMSE_FFT[n] / (doa.size * n_test))
        if is_music:
            RMSE_MUSIC[n] = np.sqrt(RMSE_MUSIC[n] / (doa.size * n_test))
        RMSE_OMP[n] = np.sqrt(RMSE_OMP[n] / (doa.size * n_test))
        if is_anm:
            RMSE_ANM[n] = np.sqrt(RMSE_ANM[n] / (doa.size * n_test))
        print(
            "SNR (dB): %.2f dB, RMSE_XAttn (deg): %.2f, RMSE_SDOA (deg): %.2f, RMSE_FFT (deg): %.2f, RMSE_OMP (deg): %.2f" % (
                SNR_dB, RMSE[n], RMSE_legacy[n], RMSE_FFT[n], RMSE_OMP[n]))

    plt.figure()
    plt.semilogy(SNR_range, RMSE, linestyle='-', marker='o', linewidth=2, markersize=8, label='SDOA-Net + Cross-Attn')
    if is_legacy:
        plt.semilogy(SNR_range, RMSE_legacy, linestyle='-', marker='D', linewidth=2, markersize=8, label='SDOA-Net (original)')
    plt.semilogy(SNR_range, RMSE_FFT, linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
    # plt.semilogy(SNR_range, RMSE_MUSIC, linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
    plt.semilogy(SNR_range, RMSE_OMP, linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
    # plt.semilogy(SNR_range, RMSE_ANM, linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (deg)')
    plt.legend()
    plt.grid()
    plt.savefig('RMSE_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    if is_save:
        io.savemat('SNR_range.mat', {'array': SNR_range})
        io.savemat('RMSE.mat', {'array': RMSE})
        io.savemat('RMSE_legacy.mat', {'array': RMSE_legacy})
        io.savemat('RMSE_FFT.mat', {'array': RMSE_FFT})
        # io.savemat('RMSE_MUSIC.mat', {'array': RMSE_MUSIC})
        io.savemat('RMSE_OMP.mat', {'array': RMSE_OMP})
        # io.savemat('RMSE_ANM.mat', {'array': RMSE_ANM})

    # plt.figure()
    # plt.semilogy(SNR_range, savitzky_golay(RMSE, 50, 3), linestyle='-', marker='o', linewidth=2, markersize=8, label='SDOA-Net + Cross-Attn')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_legacy, 50, 3), linestyle='-', marker='D', linewidth=2, markersize=8, label='SDOA-Net (original)')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_FFT, 50, 3), linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_MUSIC, 50, 3), linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_OMP, 50, 3), linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_ANM, 50, 3), linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('RMSE (deg)')
    # plt.legend()
    # plt.grid()
    # plt.show()
