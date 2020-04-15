import pandas as pd 
import numpy as np

import os
import joblib
import sklearn.metrics
import math
import cv2
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import se_resnext50
from utils import macro_recall, AugMix, seed_everything, strong_aug,  EarlyStopping

import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Cutout
)

import warnings
warnings.filterwarnings('ignore')

# Setting
HEIGHT = 137
WIDTH = 236
SIZE = 128

EPOCHS = 50
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu" 
seed_everything(314)
start_time = time.time()

'''
# code by @lafoss
'''
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


class BengaliDataset(Dataset):
    def __init__(self, csv, img_height, img_width, transform=None):
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(f'./input/train_images/{img_id}.pkl')
        img = img.reshape(self.img_height, self.img_width)
        img = 255 - img
        img = crop_resize(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        else:
            img = img

        img = img[:, :, np.newaxis]

        label_1 = self.csv.iloc[index].grapheme_root
        label_2 = self.csv.iloc[index].vowel_diacritic
        label_3 = self.csv.iloc[index].consonant_diacritic

        return (
            torch.tensor(img, dtype=torch.float).permute(2, 0, 1),
            torch.tensor(label_1, dtype=torch.long),
            torch.tensor(label_2, dtype=torch.long),
            torch.tensor(label_3, dtype=torch.long)
        )

df_train = pd.read_csv('./input/train.csv')
df_train['fold'] = pd.read_csv('./input/df_folds.csv')['fold']

trn_fold = [i for i in range(6) if i not in [5]]
vid_fold = [5]

trn_idx = df_train.loc[df_train['fold'].isin(trn_fold)].index
vid_idx = df_train.loc[df_train['fold'].isin(vid_fold)].index

# augs = [
#     GaussNoise(),
#     ShiftScaleRotate(),
#     CoarseDropout(),
#     IAAPiecewiseAffine()]

# transforms_train = albumentations.Compose([
#     AugMix(width=3, depth=2, alpha=.2, p=.5, augmentations=augs)])

trn_dataset = BengaliDataset(csv=df_train.iloc[trn_idx], img_height=HEIGHT, img_width=WIDTH, transform=strong_aug(p=0.5))
vid_dataset = BengaliDataset(csv=df_train.iloc[vid_idx], img_height=HEIGHT, img_width=WIDTH)



trn_loader = torch.utils.data.DataLoader(
    dataset = trn_dataset,
    batch_size = BATCH_SIZE,
    num_workers= 4,
    shuffle = True,
    pin_memory  = True
)

vid_loader = torch.utils.data.DataLoader(
    dataset = vid_dataset,
    batch_size = BATCH_SIZE,
    num_workers= 4,
    shuffle = False
)

model = se_resnext50().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3,verbose=True)


def train(model, train_loader, optimizer, epoch):
    model.train()
    loss = 0.0
    acc = 0.0

    final_loss = 0.0
    final_preds = []
    final_targets = []

    for idx, (img, label1, label2, label3) in enumerate(train_loader):
        img = img.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)

        optimizer.zero_grad()
        pred_grapheme, pred_vowel, pred_consonan = model(img)

        final_preds.append(torch.cat((pred_grapheme, pred_vowel, pred_consonan), dim=1))
        final_targets.append(torch.stack((label1, label2, label3), dim=1))

        loss_grapheme  = F.cross_entropy(pred_grapheme, label1)
        loss_vowel = F.cross_entropy(pred_vowel, label2)
        loss_consonan = F.cross_entropy(pred_consonan, label3)

        loss = loss_grapheme + loss_vowel + loss_consonan
        final_loss += loss

        acc += (pred_grapheme.argmax(1)==label1).float().mean()
        acc += (pred_vowel.argmax(1)==label2).float().mean()
        acc += (pred_consonan.argmax(1)==label3).float().mean()

        loss.backward()
        optimizer.step()

    final_preds = torch.cat(final_preds)
    final_targets = torch.cat(final_targets)
    recall = macro_recall(final_preds, final_targets)

    print('acc : {:.2f}% , loss : {:.4f}, Recall : {:.4f}'.format(
        acc/(len(train_loader)*3), final_loss/len(train_loader), recall))

def evaluate(model, test_loader):
    model.eval()

    final_loss = 0
    final_preds = []
    final_targets = []

    with torch.no_grad():
        for idx, (img, label1, label2, label3) in enumerate(test_loader):
            img = img.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            label3 = label3.to(device)
            
            pred_grapheme, pred_vowel, pred_consonan = model(img)

            final_preds.append(torch.cat((pred_grapheme, pred_vowel, pred_consonan), dim=1))
            final_targets.append(torch.stack((label1, label2, label3), dim=1))

            loss_grapheme  = F.cross_entropy(pred_grapheme, label1)
            loss_vowel = F.cross_entropy(pred_vowel, label2)
            loss_consonan = F.cross_entropy(pred_consonan, label3)

            loss = loss_grapheme + loss_vowel + loss_consonan
            final_loss += loss

        final_preds = torch.cat(final_preds)
        final_targets = torch.cat(final_targets)
        recall = macro_recall(final_preds, final_targets)

        return final_loss/len(test_loader), recall

def main():
    # Validation
    best_recall = -1
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(1, EPOCHS+1):
        print('Epoch {}/{} '.format(epoch, EPOCHS))
        train(model, trn_loader, optimizer, epoch)
        val_loss, val_recall = evaluate(model, vid_loader)

        early_stopping(val_recall, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(val_loss)

        if val_recall > best_recall:
                if not os.path.isdir("checkpoint"):
                    os.makedirs("checkpoint")

                print('###### Model Save, Validation Recall is : {:.4f}, Loss is : {:.4f}'.format(val_recall, val_loss))
                torch.save(model.state_dict(), './checkpoint/augmix2_seresnent50_saved_weights.pth')
                best_recall = val_recall

    print('Time to train model: {} mins'.format(round((time.time() - start_time) / 60, 2)))

if __name__ == '__main__':
    main()