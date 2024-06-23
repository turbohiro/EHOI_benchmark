import os
import pandas as pd
import logging as log
import numpy as np

from typing import Callable
from torch.utils.data import Dataset, DataLoader


def load_dataset(dataset_root):
    # base = "/dataHDD/1sliu/EhoA/hand_object/HAND_OBJECT/"
    train_file_paths = []
    test_file_paths = []
    train_labels = []
    test_labels = []
    for root, dirs, files in os.walk(dataset_root):
        if len(dirs) == 0:
            for file in files:
                label = int(os.path.basename(os.path.normpath(root + '/')))
                kind = os.path.basename(os.path.dirname(os.path.normpath(root + '/')))
                if kind == 'train':
                    train_file_paths.append(root + '/' + file)
                    train_labels.append(label)
                elif kind == 'test':
                    test_file_paths.append(root + '/' + file)
                    test_labels.append(label)
    df_test = pd.DataFrame({"label": test_labels, "file_path": test_file_paths})
    df_train = pd.DataFrame({"label": train_labels, "file_path": train_file_paths})
    log.info(f"Successful load dataset from root:{dataset_root}")
    return df_train, df_test


class CustomHandObjectDataset(Dataset):
    def __init__(self, dataframe, root, multi=True, target="pytorch", save_dir="compressed_dataset"):
        self.dataset_root = root
        self.dataframe = dataframe
        self.multi = multi
        self.target = target
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        label = self.dataframe.iloc[item]["label"]
        file_path = self.dataframe.iloc[item]["file_path"]
        data = np.load(file_path)
        input_data = data["frames"]
        input_data = np.array(input_data, dtype=np.float32)

        if self.multi:
            if self.target == "pytorch":
                return input_data, label
            else:
                return {
                    "img": input_data,
                    "label": label
                }
        else:
            relative_path = os.path.relpath(file_path, self.dataset_root)
            save_path = os.path.join(self.save_dir, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, frame=input_data, label=label)

            return save_path


def main():
    from rich.progress import track
    base = "/dataHDD/1sliu/EhoA/hand_object/frames_number_16_split_by_number/"
    train, test = load_dataset(base)
    custom_dataset = CustomHandObjectDataset(train, base)
    dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True, num_workers=4)
    print("test")
    for batch in track(dataloader):
        inputs, labels = batch
        print(inputs.shape, labels)


# if name == "main":
#     main()