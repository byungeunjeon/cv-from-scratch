import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json
import random
from scipy.misc import toimage, imsave

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from losses.loss import *
from nets.classifier import FCClassifier

from data.loader import PascalVOC
import torch.optim as optim
from utils import *

def train(dataset, model, optimizer, epoch):
    """
    TODO: Implement training for simple FC classifier.
        Input: Z-dimensional vector
        Output: label.
    """
    N = 10902
    batch_size = 32

    data_x, data_y = dataset
    data_x = torch.tensor(data_x, dtype=torch.float)
    data_y = torch.tensor(data_y, dtype=torch.long)
    dataset = data.TensorDataset(data_x, data_y)

    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # Used these small val dataset to get a sense of model and data
    # loader_train = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(0, N-256)))
    # loader_val = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(N-256, N)))
    
    model.train()

    """
    Put train loop here.
    """
    for (x, y) in loader_train:
        x = x.to('cpu')
        y = y.to('cpu')

        out = model(x)
        loss = cross_entropy1d(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch %d, loss = %.4f' % (epoch, loss.item()))
    # # val split is not required but useful to get a sense of model
    # validate(loader_val, model)
    # print()
    torch.save(model, "./models/fc_cls.pkl")


def validate(loader, model):
    """
    Not necessary for FCClassifier but helpful to get a sense of the model
    """
    print('Checking accuracy on validation set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to('cpu')
            y = y.to('cpu')
            out = model(x)
            _, predicted = torch.max(out.data, dim=1)
            num_samples += y.size(0)
            num_correct += (predicted == y).sum().item()
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def main():

    classifier = FCClassifier().cpu().float()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4) # pick an optimizer.

    dataset_x = np.load("./features/feats_x.npy")
    dataset_y = np.load("./features/feats_y.npy")

    num_epochs = 30 # your choice, try > 10

    for epoch in range(num_epochs):
        train([dataset_x, dataset_y], classifier, optimizer, epoch)

if __name__ == '__main__':
    main()
