"""
TODO: Implement zoomout feature extractor.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

class Zoomout(nn.Module):
    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and list out the layers
        self.vgg = models.vgg11(pretrained=True)
        self.feature_list = list(self.vgg.features.children())
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        """
        TODO:  load the correct layers to extract zoomout features.
        """
        self.zoomout_indices = [0, 3, 8, 13, 18] # conv layers preceding relu & maxpool


    def forward(self, x):
        """
        TODO: load the correct layers to extract zoomout features.
        Hint: use F.upsample_bilinear and then torch.cat.
        """
        # normalization as a suggestion from pytorch vgg
        x /= 225
        x = (x - self.vgg_mean) / self.vgg_std

        out = None
        for idx, feat in enumerate(self.feature_list[:19]):
            x = feat(x)
            if idx in self.zoomout_indices: 
                x_upsampled = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=True)
                if out is None:
                    out = x_upsampled
                else:
                    out = torch.cat((out, x_upsampled), dim=1)

        return out