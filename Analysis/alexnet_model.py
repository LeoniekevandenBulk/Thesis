# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:33:13 2019

@author: Leonieke

Slightly adapted Alexnet from "https://pytorch.org/docs/stable/torchvision/models.html"
to be used as a feature extractor for the analysis of the reconstructions
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, return_activations=False):
        super(AlexNet, self).__init__()
        
        self.return_activations = return_activations
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc_block1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True)
        )
        
        self.fc_block2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True) 
        )
        
        self.fc_block3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        if(self.return_activations):
            activations = []
            x = self.conv_block1(x)
            activations.append(x)
            x = self.conv_block2(x)
            activations.append(x)
            x = self.conv_block3(x)
            activations.append(x)
            x = self.conv_block4(x)
            activations.append(x)
            x = self.conv_block5(x)
            activations.append(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.fc_block1(x)
            activations.append(x)
            x = self.fc_block2(x)
            activations.append(x)
            x = self.fc_block3(x)
            activations.append(x)
            return activations
        else:
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.conv_block5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.fc_block1(x)
            x = self.fc_block2(x)
            x = self.fc_block3(x)
            return x


def alexnet(pretrained=False, progress=True, return_activations=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(return_activations=return_activations, **kwargs)
    if pretrained:
        old_state_dict = torch.load(os.path.join(os.getcwd(),'alexnet-owt-4df8aa71.pth')) #https://pytorch.org/docs/stable/torchvision/models.html
        state_dict = {}
        for i,j in zip(model.state_dict(),old_state_dict):
            state_dict[i] = old_state_dict[j]
        state_dict = OrderedDict(state_dict)
        model.load_state_dict(state_dict)
    return model