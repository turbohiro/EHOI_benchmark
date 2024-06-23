import sys

import pytorchvideo.models.resnet
import torch
import pandas as pd
from torch import nn
import os
import pytorchvideo.models.x3d
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import numpy as np

import wandb
import torch
from rich.progress import Progress

from util import load_dataset, CustomHandObjectDataset
import pytorchvideo.models

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def make_kinetics_resnet():
    n = pytorchvideo.models.resnet.create_resnet(
        input_channel=2,  # RGB input from Kinetics
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
    vit = pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers(
        spatial_size=224,
        input_channels=2,  # RGB input from Kinetics
        temporal_size = 16,  # For the tutorial let's just use a 50 layer network
        head_num_classes=8,  # Kinetics has 400 classes so we need out final head to align
    )
    csn = pytorchvideo.models.csn.create_csn(
        input_channel =2,  # RGB input from Kinetics
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
    x3d = pytorchvideo.models.x3d.create_x3d(
        input_channel =2,  # RGB input from Kinetics
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
    return x3d


# %%
#
net = make_kinetics_resnet()
# %%


#base = "/dataHDD/1sliu/EhoA/hand_object/frames_number_16_split_by_number/"
base = "/homeL/wchen/data/hand_object/frames_number_16_split_by_number"
train_data, test_data = load_dataset(base)
# Define your dataset and dataloaders (replace with your dataset and data loaders)
train_dataset = CustomHandObjectDataset(train_data, base)

val_dataset = CustomHandObjectDataset(test_data, base)
batch_size = 4
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
# Define the VideoClassifier model as shown in the previous code snippet
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# Training loop
num_epochs = 1000
# %%
# Initialize Weights and Biases
wandb.init(project="your_project_name", name="x3d_1_adam")
# Initialize a Progress instance for tracking progress
progress = Progress()
# Create a task for training progress
training_task = progress.add_task("[cyan]Training...", total=len(train_loader))
# Initialize variables to keep track of the best validation accuracy and checkpoint
best_val_accuracy = 0.0
best_checkpoint = None
# Move the model to GPU
net.to(device)
for epoch in range(num_epochs):
    net.train()  # Set the model to training mode
    total_loss = 0.0
    correct = 0
    total = 0
    with progress:
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 2, 1, 3, 4)
            #inputs = torch.nn.functional.interpolate(inputs, size=(16, 224, 224), mode='trilinear', align_corners=False)
            labels = labels.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # Update the model parameters
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update the training progress
            progress.update(training_task, advance=1)
        # scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        # Log training progress to Weights and Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy
        })
    if epoch % 5 == 0:
        # Validation loop
        net.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation during validation
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                inputs = inputs.permute(0, 2, 1, 3, 4)
                #inputs = torch.nn.functional.interpolate(inputs, size=(16, 224, 224), mode='trilinear', align_corners=False)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # scheduler.step()
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        # Log validation progress to Weights and Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy
        })
        # Check if the current validation accuracy is higher than the best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print('get points!!')
            torch.save(best_checkpoint, './best_tmp_checkpoint.pth')
# Save the best checkpoint
if best_checkpoint is not None:
    print('get points!!')
    torch.save(best_checkpoint, './best_checkpoint.pth')
# Mark the progress as completed
progress.stop()
wandb.finish()  # Finish Weights and Biases run