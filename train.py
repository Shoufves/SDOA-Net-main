import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import doasys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--numpy_seed', type=int, default=222)
    # parser.add_argument('--torch_seed', type=int, default=333)

    parser.add_argument('--n_training', type=int, default=8000, help='# of training data')  # 8000
    parser.add_argument('--n_validation', type=int, default=64, help='# of validation data')

    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=float, default=100, help='the size of grids')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')

    # module parameters
    parser.add_argument('--n_layers', type=int, default=6, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=2, help='number of filters per layer in the module')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs used to train the module')  # 100

    # array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    # parser.add_argument('--super_ratio', type=float, default=1, help='super-resolution ratio based on 102/(ant_num-1)')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--snr', type=float, default=1., help='the maximum SNR')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # imperfect parameters
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    # training policy
    parser.add_argument('--new_train', type=int, default=0, help='train a new network')
    parser.add_argument('--train_num', type=int, default=1000, help='train a new network')
    parser.add_argument('--net_type', type=int, default=0, help='the type of network')

    # 1-bit 量化开关（单比特ADC模拟）
    parser.add_argument('--is_1bit', type=int, default=0, help='enable 1-bit quantization in gen_signal (1 for on, 0 for off)')

    # ---------- Dither-CovNet ----------
    parser.add_argument('--input_mode', type=str, default='signal', choices=['signal', 'cov'],
                        help='signal: model input = raw/1-bit signal; cov: model input = dither covariance feature')
    parser.add_argument('--snapshots', type=int, default=16, help='number of snapshots L for covariance feature')
    parser.add_argument('--dither_T', type=float, default=3.0, help='uniform dither scale T')
    parser.add_argument('--snr_min_db', type=float, default=0.0, help='training SNR lower bound (dB)')
    parser.add_argument('--snr_max_db', type=float, default=30.0, help='training SNR upper bound (dB)')
    parser.add_argument('--single_stage', type=int, default=-1,
                        help='only train this train_type (0..6); -1 means run all 7 imperfect stages (quick: use 6)')
    parser.add_argument('--tag', type=str, default='', help='model filename suffix to avoid overwriting')

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

    if args.new_train:
        if args.net_type == 0:
            cov_in_dim = 2 * args.ant_num * (args.ant_num + 1) // 2  # 272 for ant_num=16
            net = doasys.spectrumModule(signal_dim=args.ant_num, n_filters=args.n_filters, inner_dim=args.inner_dim,
                                        n_layers=args.n_layers, kernel_size=args.kernel_size,
                                        in_dim=(cov_in_dim if args.input_mode == 'cov' else None))
        else:
            net = doasys.DeepFreq(signal_dim=args.ant_num, n_filters=args.n_filters, inner_dim=args.inner_dim,
                                  n_layers=args.n_layers, kernel_size=args.kernel_size,
                                  upsampling=int(args.grid_size / args.ant_num),
                                  kernel_out=int(args.grid_size / args.ant_num + 3))
    else:
        # if args.net_type == 0:
        #     net = torch.load(('net_layer%d.pkl' % args.n_layers))
        # else:
        #     net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers))

        if args.use_cuda:
            if args.net_type == 0:
                net = torch.load('net.pkl', weights_only=False)
            else:
                net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers), weights_only=False)

            # net = torch.load('net_layer2.pkl')
        else:
            if args.net_type == 0:
                net = torch.load('net.pkl', map_location=torch.device('cpu'), weights_only=False)
            else:
                net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers), map_location=torch.device('cpu'), weights_only=False)

            # net = torch.load('net_layer2.pkl', map_location=torch.device('cpu'))

    # Dither-CovNet 微调：从旧 net.pkl 加载网络，只替换第一层 in_layer（协方差输入 272 维）
    if args.input_mode == 'cov' and args.net_type == 0:
        cov_in_dim = 2 * args.ant_num * (args.ant_num + 1) // 2
        old_w = net.in_layer.weight.data
        new_in = torch.nn.Linear(cov_in_dim, old_w.shape[0], bias=False).to(old_w.device)
        # 随机初始化即可；主干权重全部保留
        net.in_layer = new_in
        print("Dither-CovNet: in_layer replaced with %d -> %d" % (cov_in_dim, old_w.shape[0]))

    # PyTorch 2.6+ compatibility: old-saved Conv1d lack _reversed_padding_repeated_twice
    import torch.nn as nn
    for module in net.modules():
        if isinstance(module, nn.Conv1d) and not hasattr(module, '_reversed_padding_repeated_twice'):
            p = module.padding
            module._reversed_padding_repeated_twice = (p, p) if isinstance(p, int) else tuple(p) * 2

    if args.use_cuda:
        net.cuda()

    # 多GPU支持：双卡3090 DataParallel
    if args.use_cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUs: %s" % (torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        net = nn.DataParallel(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)

    max_per_std = args.max_per_std
    max_amp_std = args.max_amp_std
    max_phase_std = args.max_phase_std
    max_mc = args.max_mc
    nonlinear = args.nonlinear
    is_nonlinear = args.is_nonlinear

    for idx in range(args.train_num):
        for train_type in range(7):
            if args.single_stage >= 0 and train_type != args.single_stage:
                continue
            if train_type == 0:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 1:
                args.max_per_std = max_per_std
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 2:
                args.max_per_std = 0
                args.max_amp_std = max_amp_std
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 3:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = max_phase_std
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 4:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = max_mc
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 5:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear
            else:
                args.max_per_std = max_per_std
                args.max_amp_std = max_amp_std
                args.max_phase_std = max_phase_std
                args.max_mc = max_mc
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear

            # generate the training data
            signal, doa, target_num = doasys.gen_signal(args.n_training, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            doa = torch.from_numpy(doa).float()
            ref_sp = torch.from_numpy(ref_sp).float()
            dataset = data_utils.TensorDataset(signal, ref_sp, doa)
            train_loader = data_utils.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            # generate the validation data
            signal, doa, target_num = doasys.gen_signal(args.n_validation, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            doa = torch.from_numpy(doa).float()
            ref_sp = torch.from_numpy(ref_sp).float()
            # 训练/验证统一只存干净信号，加噪/量化/协方差在 train_net 内生成
            dataset = data_utils.TensorDataset(signal, ref_sp, doa)
            val_loader = data_utils.DataLoader(dataset, batch_size=args.batch_size)

            start_epoch = 1
            criterion = torch.nn.MSELoss(reduction='sum')
            loss_train = np.zeros((args.n_epochs, 1))
            loss_val = np.zeros((args.n_epochs, 1))
            for epoch in range(start_epoch, args.n_epochs):
                net, loss_train[epoch - 1], loss_val[epoch - 1] = doasys.train_net(args=args, net=net,
                                                                                   optimizer=optimizer,
                                                                                   criterion=criterion,
                                                                                   train_loader=train_loader,
                                                                                   val_loader=val_loader,
                                                                                   doa_grid=doa_grid, epoch=epoch,
                                                                                   train_num=idx, train_type=train_type,
                                                                                   net_type=args.net_type)
                # if (epoch % 10 == 0):
                #     plt.figure()
                #     plt.semilogy(loss_train[0:epoch-1])
                #     plt.semilogy(loss_val[0:epoch-1])
                #     plt.show()
            if args.net_type == 0:
                np.savez('loss.npz' , loss_train, loss_val)
                # 多GPU时保存解包后的模型（去除DataParallel的module前缀）
                net_to_save = net.module if isinstance(net, nn.DataParallel) else net
                if args.input_mode == 'cov' or args.is_1bit:
                    if args.input_mode == 'cov':
                        fname = 'net_1bit_cov_L%d_T%s%s.pkl' % (args.snapshots, str(args.dither_T).replace('.', 'p'), args.tag)
                        torch.save(net_to_save, fname)
                        print("Dither-CovNet model saved as %s" % fname)
                    else:
                        torch.save(net_to_save, 'net_1bit.pkl')
                        print("1-bit quantized model saved as net_1bit.pkl")
                else:
                    torch.save(net_to_save, 'net.pkl')
            else:
                np.savez(('deepfreq_loss_layer%d.npz' % args.n_layers), loss_train, loss_val)
                net_to_save = net.module if isinstance(net, nn.DataParallel) else net
                torch.save(net_to_save, ('deepfreq__layer%d.pkl' % args.n_layers))
