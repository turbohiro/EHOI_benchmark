import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
from spikingjelly.activation_based import functional
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from spikingjelly.activation_based.model.srnn_scnn import  SRNN_SCNN
from spikingjelly.activation_based.model.srnn_cnn import SRNN_CNN
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import argparse
import os
import sys
import time
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Classify Hand Object')
    parser.add_argument('-model', default='srnn_scnn', type=str, help='use which model, "srnn_cnn", "srnn_crnn"')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=2, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Hand Object dataset')
    parser.add_argument('-out-dir', type=str, default='~/code/DVS/logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. sgd or adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for sgd')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-T', default=16, type=int, help='frames number')
    parser.add_argument('-C', default=8, type=int, help='class number')
    parser.add_argument('-channels', default=2, type=int, help='channel number')

    args = parser.parse_args()

    if args.model == "srnn_cnn":
        # criterion = nn.NLLLoss()#nn.CrossEntropyLoss()#
        # criterion = F.mse_loss()#nn.CrossEntropyLoss()#
        net = SRNN_CNN(None)
    elif args.model == "srnn_scnn":
        net = SRNN_SCNN(None)
    
    net.to(args.device)

    # `functional.set_step_mode` will not set neurons in LinearRecurrentContainer to use step_mode = 'm'
    functional.set_step_mode(net, step_mode='m')

    if args.cupy:
        # neurons in LinearRecurrentContainer still use step_mode = 's', so, they will still use backend = 'torch'
        functional.set_backend(net, backend='cupy')

    # Set the init constant
    frames_number = args.T
    start_epoch = 0
    max_test_acc = -1

    # Set the dataset
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=frames_number, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=frames_number, split_by='number')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b, # batch_size
        shuffle=True,
        drop_last=True,
        num_workers=args.j, # num_workers
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b, # batch_size
        shuffle=True,
        drop_last=False,
        num_workers=args.j, # num_workers
        pin_memory=True
    )
    frame,labels = next(iter(train_data_loader))
    print(f"sameple frame:{frame.shape},{labels.shape}")
    print(f"sameple label:{labels}")

    # Set the scaler
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    
    # Set the optimizer
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    # Set the scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'DVSHandObjectSRNN_SCNN_IF_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')
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

    # Trainning
    if args.resume:
        # Test
        net.eval()

        # 注册钩子
        output_layer = net.layer4[1].sn2 # 输出层
        output_layer.v_seq = []
        output_layer.s_seq = []
        def save_hook(m, x, y):
            m.v_seq.append(m.v.unsqueeze(0))
            m.s_seq.append(y.unsqueeze(0))

        output_layer.register_forward_hook(save_hook)
        with torch.no_grad():
            for frame, label in train_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                out_fr = net(frame).mean(0)
                out_spikes_counter_frequency = (out_fr ).cpu().numpy()
                print(f'Firing rate: {out_spikes_counter_frequency}')

                output_layer.v_seq = torch.cat(output_layer.v_seq)
                output_layer.s_seq = torch.cat(output_layer.s_seq)
                v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
                np.save("v_t_array.npy",v_t_array)
                s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
                np.save("s_t_array.npy",s_t_array)
                
    else:
        for epoch in tqdm(range(start_epoch, args.epochs)):
            # Train
            start_time = time.time()
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in tqdm(train_data_loader):
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 8).float()

                if scaler is not None:
                    with amp.autocast():
                        out_fr,loss,_ = net(frame,labels=label_onehot)
                        # out_fr = net_result[0] # [1] is not using
                        # out_fr = out_fr.mean(1) # [2,16,8] -> [2,8]
                        # loss = F.mse_loss(out_fr, label_onehot)
                        out_fr = out_fr.mean(0)
                        #print(f"out_fr.shape:{out_fr.shape}")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                    loss.backward()
                    optimizer.step()

                train_samples += label.numel()
                #train_samples += len(label_onehot)
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(0) == label).float().sum().item()
                functional.reset_net(net)

            train_time = time.time()
            train_speed = train_samples / (train_time - start_time)
            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            # Valid 
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
                    out_fr,loss,_ = net(frame,labels=label_onehot)
                    # out_fr = net(frame).mean(0)
                    # loss = F.mse_loss(out_fr, label_onehot)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(0) == label).float().sum().item()
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
            # print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
    

if __name__ == '__main__':
    main()