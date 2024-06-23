from dv import AedatFile
import cv2

#%%
# file_name = '/media/wchen/TRANSCEND/mugdata_light/mug1.aedat4'
# with AedatFile(file_name) as f:
#     # list all the names of streams in the file
#     print(f.names)

#     # Access dimensions of the event stream
#     height, width = f['events'].size  #260*346

#     # loop through the "events" stream
#     for e in f['events']:
#         print(e.timestamp)#1674295582506812
#     # loop through the "frames" stream
#     for frame in f['frames']:
#         print(frame.timestamp)
#         cv2.imshow('out', frame.image)
#         cv2.waitKey(1)
#%%
# from typing import Callable, Dict, Optional, Tuple
# import numpy as np
# from torchvision.datasets.utils import extract_archive
# import os
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor
# import time
# from dv import AedatFile
# import pandas as pd
# import numpy.lib.recfunctions as rfn
# np_savez = np.savez_compressed
# def load_origin_data(file_name: str) -> Dict:
#         '''
#         :param file_name: path of the events file
#         :type file_name: str
#         :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
#         :rtype: Dict

#         This function defines how to read the origin binary data.
#         '''
#         with AedatFile(file_name) as f:
#             # events will be a named numpy array
#             events = np.hstack([packet for packet in f['events'].numpy()])
#             events = rfn.rename_fields(events,{'timestamp':'t','polarity':'p'})
#             # Access information of all events by type
#             # timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
#             # # Access individual events information
#             # event_123_x = events[123]['x']
#             # # Slice events
#             # first_100_events = events[:100]
#             return events
        
# def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
#     events = load_origin_data(aedat_file)
#     print(f'Start to split [{aedat_file}] to samples.')
#     # read csv file and get time stamp and label of each sample
#     # then split the origin data to samples
#     csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)
#     #csv_data = pd.read_csv(csv_file)
#     # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
#     label_file_num = [0] * 2
#     # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
#     # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
#     # are replaced by new codes.
#     #for label in range(2):
#     #    os.mkdir(os.path.join(output_dir, str(label)))

#     length = int(csv_data.shape[0]/2)
#     for i in range(length):
#         for j in range(2):
#         # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
#             label_ori =  csv_data[2*i+j][1] -1

#             label = csv_data[2*i+j][1] +3
#             t_start = csv_data[2*i+j][2] + events['t'].min()
#             t_end = csv_data[2*i+j][3] +events['t'].min()
#             mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
#             file_name = os.path.join(output_dir, str(label), f'{fname}_{2*i+j}_{label_file_num[label_ori]}.npz')
#             np_savez(file_name,
#                         t=events['t'][mask],
#                         x=events['x'][mask],
#                         y=events['y'][mask],
#                         p=events['p'][mask]
#                         )
#             print(f'[{file_name}] saved.')

#             label_file_num[label_ori] += 1

# fname = 'bottle_user1_light'
# aedat_dile = '/data/event_camera/hand_object_original_data/bottle/bottle_light/bottle1.aedat4'
# csv_file = '/data/event_camera/hand_object_original_data/bottle/bottle_light/bottle1_label.csv'
# output_dir = '/data/event_camera/SNN/data/hand_object/events_np/train'

# a = split_aedat_files_to_np(fname, aedat_dile, csv_file, output_dir)
#%% show events
# from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
# root_dir = '/data/event_camera/SNN/data/hand_object'

# # # #%%
# train_set = DVS128Gesture(root_dir, train=True, data_type='event') #['t', 'x', 'y', 'p']--->[80048267,49,92,0]
# test_set = DVS128Gesture(root_dir, train=False, data_type='event') #[F,C,H,W]-->[20,2,128,128]
# event, label = train_set[0]
# for k in event.keys():
#     print(k, event[k])
# print('label', label)
# import pdb
# pdb.set_trace()
#%%
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import numpy as np
from typing import Callable, Dict, Optional, Tuple
root_dir = '/media/wchen/linHDD/wchen/hand_object/ '

def cal_fixed_frames_number_segment_index(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) // frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r

def integrate_events_segment_to_frame(x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
    # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    # We must use ``bincount`` rather than simply ``+``. See the following reference:
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments

    # Here is an example:

    # height = 3
    # width = 3
    # frames = np.zeros(shape=[2, height, width])
    # events = {
    #     'x': np.asarray([1, 2, 1, 1]),
    #     'y': np.asarray([1, 1, 1, 2]),
    #     'p': np.asarray([0, 1, 0, 1])
    # }
    #
    # frames[0, events['y'], events['x']] += (1 - events['p'])
    # frames[1, events['y'], events['x']] += events['p']
    # print('wrong accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # for i in range(events['p'].__len__()):
    #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
    # print('correct accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # frames = frames.reshape(2, -1)
    #
    # mask = [events['p'] == 0]
    # mask.append(np.logical_not(mask[0]))
    # for i in range(2):
    #     position = events['y'][mask[i]] * width + events['x'][mask[i]]
    #     events_number_per_pos = np.bincount(position)
    #     idx = np.arange(events_number_per_pos.size)
    #     frames[i][idx] += events_number_per_pos
    # frames = frames.reshape(2, height, width)
    # print('correct accumulation by bincount\n', frames)

    frame = np.zeros(shape=[2, H * W])
    x = x[j_l: j_r].astype(int)  # avoid overflow
    y = y[j_l: j_r].astype(int)
    p = p[j_l: j_r]
    mask = []
    mask.append(p == 0)
    mask.append(np.logical_not(mask[0]))
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
        
    return frame.reshape((2, H, W))

def integrate_events_by_fixed_frames_number(events: Dict, split_by: str, frames_num: int, H: int, W: int) -> np.ndarray:
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
    j_l, j_r = cal_fixed_frames_number_segment_index(t, split_by, frames_num)
    frames = np.zeros([frames_num, 2, H, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])

    return frames
import os

event_dir = '/media/wchen/linHDD/wchen/hand_object/events_np/train/6'
for file in os.listdir(event_dir):

    events = np.load(os.path.join(event_dir, file))

    frame = integrate_events_by_fixed_frames_number(events,frames_num=16, split_by='number',H=260,W = 346)
    # train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=20, split_by='number')
    # frame, label = train_set[0]
    print(frame.shape)

    from spikingjelly.datasets import play_frame
    # frame, label = train_set[500]
    output_dir = '/data/event_camera/SNN/visualization/6/train/'
    play_frame(frame,os.path.join(output_dir, file)+'.gif')
#%%
# from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
# root_dir = '/data/event_camera/SNN/data/hand_object'

# # # #%%
# train_set = DVS128Gesture(root_dir, train=True, data_type='frame', frames_number=20, split_by='number')
# test_set = DVS128Gesture(root_dir, train=False, data_type='frame', frames_number=20, split_by='number')
# import pdb
# pdb.set_trace()