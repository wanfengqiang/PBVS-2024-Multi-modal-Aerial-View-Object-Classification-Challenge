import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tqdm import tqdm
import os
from collections import OrderedDict
from itertools import cycle
import cv2
import pdb
from pdb import set_trace as bp
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor
import itertools
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from typing import Sequence

torch.autograd.set_detect_anomaly(True)
device_ids=[3]
device = f'cuda:{device_ids[0]}'

map_name = ['sedan', 'SUV', 'pickup_truck', 'van', 'box_truck','motorcycle','flatbed_truck','bus','pickup_truck_w_trailer','semi_w_trailer']
map_name2id = {}
for idx, name in enumerate(map_name):
    map_name2id[name] = idx
class ImageSet(Dataset):
    def __init__(self, datacsv, datapath, transform):
        self.info = pd.read_csv(datacsv)
        self.datapath = datapath
        self.data_dict = {'sedan':  0,'SUV': 1, 'pickup_truck': 2,'van': 3, 'box_truck': 4,'motorcycle': 5, 'flatbed_truck': 6, 'bus': 7,'pickup_truck_w_trailer': 8, 'semi_w_trailer': 9}
        self.trans = transform
    
    def __len__(self):
        return len(self.info)
    
    
    def __getitem__(self, index):
        label = self.data_dict[self.info["label"][index]]
        sarImage = self.trans(Image.open(self.datapath+ self.info["label"][index]+"/"+self.info["image_id"][index]).convert("RGB"))
        return sarImage, label

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

transform = transforms.Compose(
        [transforms.Resize(224), 
         transforms.ToTensor(), 
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         ])

class ImageFolderWithNames(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path_sar, target = self.samples[index]
        image_sar = self.loader(path_sar)
        if self.transform is not None:
            image_sar = self.transform(image_sar)
        folder_name = self.imgs[index][0].split('/')[-2]
        return image_sar, map_name2id[folder_name]
train_set = ImageFolderWithNames('./datasets/SAR', transform)
from torchsampler import ImbalancedDatasetSampler
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, sampler=ImbalancedDatasetSampler(train_set),
    num_workers=8, pin_memory=True, drop_last=False)

num_classes = 10

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def train():
    model_SAR = torch.load('./weights/results/SAR_feature_5000.pth', map_location = device)

    model_SAR.to(device)
    criterion = FocalLoss(gamma=0.5)

    optim_my = optim.Adam(params = model_SAR.parameters(), lr = 1e-4)
    all_epochs = 50
    scheduler = CosineAnnealingLR(optim_my, T_max=all_epochs)
    total_iter = 0
    for epoch in range(all_epochs):
        model_SAR.train()
        train_loss_SAR = 0.0
        correct_SAR = 0.0
        total_SAR = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            total_iter += 1
            inputs_SAR, labels = data[0], data[1]
            inputs_SAR, labels = inputs_SAR.to(device), labels.to(device)

            outputs_SAR = model_SAR(inputs_SAR)
            loss_SAR = criterion(outputs_SAR, labels)
            loss = loss_SAR
            
            optim_my.zero_grad()
            loss.backward()
            optim_my.step()
            
            predictions_SAR = outputs_SAR.argmax(dim=1, keepdim=True).squeeze()
            correct_SAR += (predictions_SAR == labels).sum().item()
            total_SAR += labels.size(0)
            
            train_loss_SAR += loss.item()

            if total_iter % 50 == 0:
                torch.save({
                    'model' : model_SAR,
                    }, f'./weights/results/SAR_final_{total_iter}.pth')
        accuracy_SAR = correct_SAR / total_SAR
        scheduler.step()
        print('Loss_SAR after epoch {:} is {:.2f} and accuracy_SAR is {:.2f}'.format(epoch,(train_loss_SAR / len(train_loader)),100.0*accuracy_SAR))

        
    print('Finished Pre-training')
    print()

if __name__ == "__main__":
    train()

