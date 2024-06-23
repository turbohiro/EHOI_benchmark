import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net,spiking_resnet,spiking_vgg
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
from koila import LazyTensor, lazy
import numpy as np
import torchvision
from spikingjelly import visualizing
import matplotlib.pyplot as plt
from itertools import cycle, islice
from PIL import Image

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8

    parser = argparse.ArgumentParser(description='Classify hand object interaction')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', type=str, help='root dir of save feature map')
    args = parser.parse_args()
    print(args)

    #net = parametric_lif_net.DVSHandObjectSRNNNet(channels=args.channels, spiking_neuron=neuron.ParametricLIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    #net = spiking_resnet.spiking_resnet34(pretrained=False, spiking_neuron=neuron.ParametricLIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    #net = spiking_resnet.SpikingResNetAttention(pretrained=False, spiking_neuron=neuron.ParametricLIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    #net = parametric_lif_net.DVSHandObjectNet(channels=args.channels, spiking_neuron=neuron.ParametricLIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    print(net)
    torch.cuda.empty_cache()


    net.to(args.device)
    #fixed frame-based
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    #time surface based
    # train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', duration = 1000000)
    # test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', duration = 1000000)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'DVSHandObject_FB_PLIF_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
    if args.resume:
         # 保存绘图用数据
        net.eval()
        # Visualize feature maps
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        net.sn1.register_forward_hook(get_activation('sn1'))

        # 注册钩子for 3d heatmap and 1d spike figures
        # output_layer = net.layer4[1].sn2 # 输出层
        # output_layer.v_seq = []
        # output_layer.s_seq = []
        # def save_hook(m, x, y):
        #     m.v_seq.append(m.v.unsqueeze(0))
        #     m.s_seq.append(y.unsqueeze(0))
        # output_layer.register_forward_hook(save_hook)
        y_pred = []
        y_true = []
        x_image = []
        
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                label = label.to(args.device)
                # x_image.extend(np.reshape(frame.data.cpu().numpy(),(16,179920)))
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                start = time.time()
                out_fr = net(frame).mean(0)
                finish = time.time()
                # y_pred.extend(out_fr.argmax(1).data.cpu().numpy()) # Save Prediction
                # #y_true.extend(label.data.cpu().numpy()) # Save Prediction
                # y_true.extend(list(islice(cycle(label.data.cpu().numpy()), 16)))
                # spike_seq = activation['sn1']
                # to_pil_img = torchvision.transforms.ToPILImage()
                # vs_dir = os.path.join(args.save_es, 'visualization')
                # #os.mkdir(vs_dir)
                # img = frame[0].cpu()
                # spike_seq = spike_seq.cpu()
                # for i in range(label.shape[0]):
                #     vs_dir_i = os.path.join(vs_dir, f'{i}')
                #     #os.mkdir(vs_dir_i)
                #     out = to_pil_img(img[i])
                #     cm = plt.get_cmap('tab20c')
                #     colored_image = cm(np.array(out)[:,:,0])
                #     Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(vs_dir_i, f'input.png'))
                #     #out.save(os.path.join(vs_dir_i, f'input.png'))
                #     for t in range(16):
                #         print(f'saving {i}-th sample with t={t}...')
                #         # spike_seq.shape = [T, N, C, H, W]
                #         visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                #         plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'))
                #         #plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'))
                #         #plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'))
                #         plt.clf()
                # print test result
                print()
                print("Time consumed:", finish - start)
                print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
                import pdb
                pdb.set_trace()
                # ##used for plot 3d heatmap and 1d spike figures
                # out_spikes_counter_frequency = (out_fr ).cpu().numpy()
                # print(f'Firing rate: {out_spikes_counter_frequency}')

                # output_layer.v_seq = torch.cat(output_layer.v_seq)
                # output_layer.s_seq = torch.cat(output_layer.s_seq)
                # v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
                # np.save("v_t_array.npy",v_t_array)  
                # s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
                # np.save("s_t_array.npy",s_t_array)
        
            # constant for classes
            classes = ('Drink', 'Contain', 'Pour', 'Wrap', 'Handover',
                    'Screw', 'Tooluse', 'Grasp')
            
            from sklearn.metrics import confusion_matrix
            import pandas as pd
            import seaborn as sn
            #PCA
            from sklearn.manifold import TSNE
            # tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(np.array(x_image))
            # #fashion_tsne = TSNE(random_state=99).fit_transform(np.array(x_image))
            # #fashion_scatter(fashion_tsne,y_true)
                      
            # plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y_true, cmap='Spectral')
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.colorbar(boundaries=np.arange(9)-0.5).set_ticks(np.arange(8))
            # #plt.title('Visualizing Kannada MNIST through t-SNE', fontsize=24)
            # plt.savefig('sne2.pdf')

            #TSNE(3)
            tsne = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=300).fit_transform(np.array(x_image))
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 1], s= 5, c=y_true, cmap='Spectral')
            #plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(9)-0.5).set_ticks(np.arange(8))
            plt.savefig('sne3_v2.pdf')
            import pdb
            pdb.set_trace()

            # Build confusion matrix
            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                                columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, cmap="Blues",annot=True) #Greens
            plt.savefig('output2.pdf')
               
    else:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 8).float()


                if scaler is not None:
                    with amp.autocast():
                        out_fr = net(frame).mean(0)
                        loss = F.mse_loss(out_fr, label_onehot)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    loss.backward()
                    optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net)

            train_time = time.time()
            train_speed = train_samples / (train_time - start_time)
            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            net.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(args.device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    label = label.to(args.device)
                    label_onehot = F.one_hot(label, 8).float()
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True

            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

            print(args)
            print(out_dir)
            print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
            print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
            print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
   
    main()