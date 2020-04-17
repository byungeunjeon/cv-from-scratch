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

from losses.loss import *
from nets.zoomout import Zoomout
from nets.classifier import FCClassifier, DenseClassifier

from data.loader import *
import torch.optim as optim
from utils import *

def train(args, zoomout, model, train_loader, optimizer, epoch):
    count = 0

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        """
        TODO: Implement training loop.
        """
        images = images.to('cpu')
        labels = labels.to('cpu')

        with torch.no_grad(): # zoomout should not involve gradient training
            x = zoomout(images.cpu().float())

        predicts = model(x) # DenseClassifier
        loss = cross_entropy2d(predicts, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            count = count + 1
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.item()))

        if batch_idx % 100 == 0: # changed from 20 to 100
            """
            Visualization of results.
            """
            pred = predicts[0,:,:,:]
            gt = labels[0,:,:].data.numpy().squeeze()
            im = images[0,:,:,:].data.numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.data.numpy().squeeze()
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')

            image.save("./imgs/train/im_" + str(count) + "_" + str(epoch) + "_.png")
            visualize("./lbls/train/pred_" + str(count) + "_" + str(epoch) + ".png", pred)
            visualize("./lbls/train/gt_" + str(count) + "_" + str(epoch) + ".png", gt)

    # Make sure to save your model periodically
    torch.save(model, "./models/full_model.pkl")

def val(args, zoomout, model, val_loader, epoch):
    # modified from https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/evaluate.py
    model.eval()
    print("Validating...")
    label_trues, label_preds = [], []
    count = 0

    for batch_idx, (data, target) in enumerate(val_loader):

        data, target, im_viz, lbl_viz = data.float(), target.float(), data.float(), target.to('cpu')
        score = model(zoomout(data))

        _, pred = torch.max(score, dim=1) # changed dim from 0 to 1
        lbl_pred = pred.data.numpy().astype(np.int64)
        lbl_true = target.data.numpy().astype(np.int64)

        for _, lt, lp in zip(_, lbl_true, lbl_pred):
            label_trues.append(lt)
            label_preds.append(lp)

        if (batch_idx % 10 == 0) and (epoch in [0, 1, 4, 9]):
            count = count + 1
            """
            Visualization of results on val dataset. 
            epoch 0: only with weights learned from FCClassifier
            epoch 1: after 2 epochs of training DenseClassifier
            epoch 4: after 5 epochs of training DenseClassifier
            epoch 9: after 10 epochs of training DenseClassifier
            """
            pred = score[0,:,:,:]
            gt = lbl_viz[0,:,:].data.numpy().squeeze()
            im = im_viz[0,:,:,:].data.numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.data.numpy().squeeze()
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')

            image.save("./imgs/val/im_" + str(count) + "_" + str(epoch) + "_.png")
            visualize("./lbls/val/pred_" + str(count) + "_" + str(epoch) + ".png", pred)
            visualize("./lbls/val/gt_" + str(count) + "_" + str(epoch) + ".png", gt)

    n_class = 21
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))


def main():
    # You can add any args you want here
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models/zoomoutscratch_pascal_1_6.pkl', help='Path to the saved model')
    parser.add_argument('--pretrained_viz', nargs='?', type=bool, default=False, help='Viz using saved model, No training')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,  help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, help='Learning Rate')

    args = parser.parse_args()

    zoomout = Zoomout().float()

    # we will not train the feature extractor
    for param in zoomout.parameters():
        param.requires_grad = False

    fc_classifier = torch.load('./models/fc_cls.pkl')
    classifier = DenseClassifier(fc_model=fc_classifier).float()

    """
       TODO: Pick an optimizer.
       Reasonable optimizer: Adam with learning rate 1e-4.  Start in range [1e-3, 1e-4].
    """
    optimizer = optim.SGD(classifier.parameters(), lr=args.l_rate)

    dataset_train = PascalVOC(split = 'train')
    dataset_val = PascalVOC(split = 'val')

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    if args.pretrained_viz == True:
        print('Vizualization of val dataset on pretrained model...')
        classifier = torch.load(args.model_path)
        val(args, zoomout, classifier, val_loader, epoch=1)
    else: 
        for epoch in range(args.n_epoch):
            if epoch == 0:
                val(args, zoomout, classifier, val_loader, epoch) # directly using weights without fine-tuning
                train(args, zoomout, classifier, train_loader, optimizer, epoch)
                # val(args, zoomout, classifier, val_loader, epoch) # skip for the first epoch
            else:
                train(args, zoomout, classifier, train_loader, optimizer, epoch)
                val(args, zoomout, classifier, val_loader, epoch)

if __name__ == '__main__':
    main()
