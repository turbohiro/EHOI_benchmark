from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


#1. build event datatset--processing
# root_dir = '/homeL/wchen/SNNs-RNNs/data'
# train_set = DVS128Gesture(root_dir, train=True,data_type='event')
# test_set = DVS128Gesture(root_dir, train=True, data_type='event')

# event, label = train_set[0]
# for k in event.keys():
#     print(k,event[k]) #['t', 'x', 'y', 'p']
# print('label',label)

#2. build frame datatset--processing
# root_dir = '/homeL/wchen/SNNs-RNNs/data'
# train_set = DVS128Gesture(root_dir, train=True,data_type='frame',frames_number = 20, split_by = 'number')
# test_set = DVS128Gesture(root_dir, train=True, data_type='frame',frames_number = 20, split_by = 'number')

# frame, label = train_set[0] #(20, 2, 128, 128)
# print(frame.shape)

# from spikingjelly.datasets import play_frame
# frame, label = train_set[0]
# play_frame(frame,save_gif_to=True)

#%%3. constant timesteps frame dataset--processing
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask, dvs128_gesture
root='D:/datasets/DVS128Gesture'
train_set = dvs128_gesture.DVS128Gesture(root, data_type='frame', duration=1000000, train=True)
for i in range(5):
    x, y = train_set[i]
    print(f'x[{i}].shape=[T, C, H, W]={x.shape}')
train_data_loader = DataLoader(train_set, collate_fn=pad_sequence_collate, batch_size=5)
for x, y, x_len in train_data_loader:
    print(f'x.shape=[N, T, C, H, W]={tuple(x.shape)}')
    print(f'x_len={x_len}')
    mask = padded_sequence_mask(x_len)  # mask.shape = [T, N]
    print(f'mask=\n{mask.t().int()}')
    break

train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)
from spikingjelly.datasets import play_frame
frame, label = train_set[500]
play_frame(frame)