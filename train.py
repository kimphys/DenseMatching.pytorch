import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

global param_e
is_cuda = torch.cuda.is_available()

class MyDataset_npy(Dataset):
    def __init__(self, np_img_path, np_lbl_path):
        self.img_files = list(np.load(np_img_path))
        self.lbl_files = list(np.load(np_lbl_path))
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.img_files[index]).float() # H, W, C
        img  = img.permute(2, 0, 1) # H, W, C -> C, H, W
        
        lbl = torch.tensor(self.lbl_files[index]).float()
        lbl = np.expand_dims(lbl,axis=0)

        return img, lbl

    def __len__(self):
        return len(self.img_files)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DenseBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=in_channels + 0 * middle_channels,out_channels=middle_channels,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(in_channels=in_channels + 1 * middle_channels,out_channels=middle_channels,kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(in_channels=in_channels + 2 * middle_channels,out_channels=middle_channels,kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(in_channels=in_channels + 3 * middle_channels,out_channels=middle_channels,kernel_size=3,padding=1)
        self.conv_5 = nn.Conv2d(in_channels=in_channels + 4 * middle_channels,out_channels=out_channels,kernel_size=1)

    def forward(self, x):
        x0 = x

        x1 = self.relu(self.conv_1(x0))
        c1_dense = self.relu(torch.cat([x0, x1], 1))

        x2 = self.relu(self.conv_2(c1_dense))
        c2_dense = self.relu(torch.cat([x0, x1, x2], 1))

        x3 = self.relu(self.conv_3(c2_dense))
        c3_dense = self.relu(torch.cat([x0, x1, x2, x3], 1))

        x4 = self.relu(self.conv_4(c3_dense))
        c4_dense = self.relu(torch.cat([x0, x1, x2, x3, x4], 1))

        x5 = self.relu(self.conv_5(c4_dense))

        return x5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.module_list = Dense_connectivity_layers()

    def forward(self, x):

        for module in self.module_list:
            name = module.__dir__
            x = module(x)
        
        flat_size = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, flat_size)

        # Fully-connected layer 1
        x = nn.Linear(flat_size, 256)(x)
        x = nn.ReLU(inplace=True)(x)

        # Fully-connected layer 2
        x = nn.Linear(256, 1)(x)
        x = nn.Sigmoid()(x)

        return x

def Dense_connectivity_layers():
    out_filters = [4] # input_channels = 3 (RGB) + 1 (IR)

    module_list = nn.ModuleList()

    ## Convolutional operation + MaxPool2d
    modules = nn.Sequential()
    filters = 64
    modules.add_module('Conv2d_1',nn.Conv2d(in_channels=out_filters[-1], out_channels=filters, kernel_size=3, padding=1))
    modules.add_module('activation_1',nn.ReLU(inplace = True))
    modules.add_module('MaxPool2d_1',nn.MaxPool2d(kernel_size=2, stride=2))
    module_list.append(modules)
    out_filters.append(filters)

    ## Densely-connected block + MaxPool2d
    modules = nn.Sequential()
    mid_filters = 64
    filters = 256
    modules.add_module('DenseBlock_1', DenseBlock(in_channels=out_filters[-1],middle_channels=mid_filters,out_channels=filters))
    modules.add_module('MaxPool2d_2',nn.MaxPool2d(kernel_size=2, stride=2))
    module_list.append(modules)
    out_filters.append(filters)

    ## Convolution operation
    modules = nn.Sequential()
    filters = 256
    modules.add_module('Conv2d_2',nn.Conv2d(in_channels=out_filters[-1], out_channels=filters, kernel_size=3, padding=1))
    modules.add_module('activation_2',nn.ReLU(inplace=True))
    modules.add_module('Conv2d_3',nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1))
    modules.add_module('activation_3',nn.ReLU(inplace=True))
    module_list.append(modules)
    out_filters.append(filters)

    return module_list
    
def compute_loss(p, targets):
    global param_e
    e = param_e

    loss1 = F.binary_cross_entropy(p, targets)
    loss2 = F.binary_cross_entropy(p, torch.ones_like(p) / 2)

    return (1 - e) * loss1 + e * loss2


if __name__ == '__main__':
    global param_e
    param_e = 0.05

    model = Net()

    learning_rate = 1e-3
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    np_img_path = './test_data/test_images.npy'
    np_lbl_path = './test_data/test_labels.npy'

    trainloader = DataLoader(MyDataset_npy(np_img_path, np_lbl_path), batch_size=2, shuffle=True, num_workers=1)

    if is_cuda:
        model.cuda()

    n_epoch = 30

    for epoch in range(n_epoch):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):

            print(inputs.shape, labels.shape)

            if is_cuda:
                inputs, labels =inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            predicts = model(inputs)
            loss = compute_loss(predicts, labels)
            loss.backward()
            optimizer.step()

    print('Finished training')    