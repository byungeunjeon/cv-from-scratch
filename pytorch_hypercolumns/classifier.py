"""
   Here you will implement a relatively shallow neural net classifier on top of the hypercolumn (zoomout) features.
   You can look at a sample MNIST classifier here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from .zoomout import *
import numpy as np
from torchvision import transforms

class FCClassifier(nn.Module):
    """
        Fully connected classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, n_classes=21):
        super(FCClassifier, self).__init__()
        """
        TODO: Implement a fully connected classifier.
        """
        # You will need to compute these and store as *.npy files
        self.mean = torch.Tensor(np.load("./features/mean_sample.npy"))
        self.std = torch.Tensor(np.load("./features/std_sample.npy"))
        self.fc1 = nn.Linear(1472, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.5) # Dropout here is optional but helpful. See writeup for results of experiment
        self.fc3 = nn.Linear(256, 21)

    def forward(self, x):
        # normalization
        x = (x - self.mean)/self.std
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x) # Dropout here is optional but helpful. See writeup for results of experiment
        x = self.fc3(x)
        return x


class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()
        """
        TODO: Convert a fully connected classifier to 1x1 convolutional.
        """

        self.mean = torch.Tensor(np.load("./features/mean_global.npy"))
        self.std = torch.Tensor(np.load("./features/std_global.npy"))
        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(self.mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(self.std, -1), -1))

        self.conv1 = nn.Conv2d(1472, 256, 1)
        self.conv2 = nn.Conv2d(256, 256, 1)
        self.conv3 = nn.Conv2d(256, 21, 1)

        self.fc_transfer = fc_model.state_dict()
        self.conv1.weight.data = self.fc_transfer['fc1.weight'].view(256, 1472, 1, 1)
        self.conv2.weight.data = self.fc_transfer['fc2.weight'].view(256, 256, 1, 1)
        self.conv3.weight.data = self.fc_transfer['fc3.weight'].view(21, 256, 1, 1)

    def forward(self, x):
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """
        # normalization
        x = (x - self.mean)/self.std
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x_upsampled = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        return x_upsampled
