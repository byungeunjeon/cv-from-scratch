import sys
import torch
import numpy as np

from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
import gc

def extract_samples(zoomout, dataset):
    """
    TODO: Follow the directions in the README
    to extract a dataset of 1x1xZ features along with their labels.
    Predict from zoomout using:
         with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
    """
    # Initialize tensors/variables
    features = None # will be replaced by first tensor, later torch.cat the results
    features_labels = None # will be replaced by first tensor, later torch.cat the results
    running_E_X = torch.zeros((1, 1472)) # will sum the values seen so far
    running_E_X2 = torch.zeros((1, 1472)) # will sum the values^2 seen so far

    for image_idx in range(len(dataset)):
        images, labels = dataset[image_idx]
        with torch.no_grad(): # zoomout should not involve gradient training
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))

        running_mean = zoom_feats.view(1472, 112*112).mean(dim=1).view(1, -1)
        running_E_X += running_mean
        running_E_X2 += running_mean ** 2

        unique_labels = labels.unique() # for samples-per-class
        if features is None:
            features_labels = unique_labels.repeat(3,1).transpose(0,1).reshape(-1)
        else:
            labels_to_cat = (features_labels, unique_labels.repeat(3,1).transpose(0,1).reshape(-1))
            features_labels = torch.cat(labels_to_cat, dim=0)

        sampled_idx = None # will be replaced by first tensor, later torch.cat the results
        for i in range(len(unique_labels)):
            y = unique_labels[i]
            y_idx = (labels == y).nonzero() # shape: (N_label_is_y, 2)
            rnd = torch.randint(0, y_idx.size(0), (3,)) # randomly sampled 3 idx of y_idx
            if sampled_idx is None:
                sampled_idx = y_idx[rnd] # shape: (3, 2)
            else:
                sampled_idx = torch.cat((sampled_idx, y_idx[rnd]), dim=0)

        for j in range(sampled_idx.size(0)):
            rc = sampled_idx[j].view(-1)
            r, c = rc[0]//2, rc[1]//2 # features are half the input resolution
            z = zoom_feats[:, :, r, c].view(1, -1) # shape: (1, 1472)
            if features is None:
                features = z
            else:
                features = torch.cat((features, z), dim=0)

    mean = running_E_X / len(dataset)
    E_X2 = running_E_X2 / len(dataset)
    std = (E_X2 - mean ** 2).sqrt() # E(X^2) - [E(X)]^2
    return features, features_labels, mean, std


def main():
    zoomout = Zoomout().cpu().float()
    for param in zoomout.parameters():
        param.requires_grad = False

    dataset_train = PascalVOC(split = 'train')

    features, labels, mean, std = extract_samples(zoomout, dataset_train)

    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)
    np.save("./features/mean_global.npy", mean)
    np.save("./features/std_global.npy", std)

    feats_mean = features.mean(dim=0).view(1, 1472)
    feats_std = features.std(dim=0).view(1, 1472)
    np.save("./features/mean_sample.npy", feats_mean)
    np.save("./features/std_sample.npy", feats_std)


if __name__ == '__main__':
    main()
