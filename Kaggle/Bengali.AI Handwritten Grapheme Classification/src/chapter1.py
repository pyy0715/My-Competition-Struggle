import pandas as pd 
import numpy as np
import sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')

from utils import macro_recall, AugMix

import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import HorizontalFlip
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CoarseDropout,
    GaussNoise,  GaussianBlur, IAAPiecewiseAffine, RandomContrast, RandomBrightness
)


import cv2
import time
import random
import os
import matplotlib.pyplot as plt
import joblib

HEIGHT = 137
WIDTH = 236
SIZE = 128

SEED = 715
EPOCHS = 50
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu" 

start_time = time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

seed_everything(SEED)

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
        img = (img*(255.0/img.max())).astype(np.uint8)
        img = crop_resize(img)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
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

augs = [
    HorizontalFlip(always_apply=True),
    GaussNoise(always_apply=True),
    ShiftScaleRotate(rotate_limit=20, always_apply=True),
    RandomContrast(always_apply=True),
    RandomBrightness(always_apply=True),
    CoarseDropout(always_apply=True),
    IAAPiecewiseAffine(always_apply=True)]

transforms_train = albumentations.Compose([
    AugMix(width=3, depth=2, alpha=.2, p=.5, augmentations=augs)])

trn_dataset = BengaliDataset(csv=df_train.iloc[trn_idx], img_height=HEIGHT, img_width=WIDTH, transform=transforms_train)
vid_dataset = BengaliDataset(csv=df_train.iloc[vid_idx], img_height=HEIGHT, img_width=WIDTH)

trn_loader = torch.utils.data.DataLoader(
    dataset = trn_dataset,
    batch_size = BATCH_SIZE,
    num_workers= 5,
    shuffle = True
)

vid_loader = torch.utils.data.DataLoader(
    dataset = vid_dataset,
    batch_size = BATCH_SIZE,
    num_workers= 5,
    shuffle = False
)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        
        # grapheme_root
        self.fc1 = nn.Linear(2048, 168)
        # vowel_diacritic
        self.fc2 = nn.Linear(2048, 11)
        # consonant_diacritic
        self.fc3 = nn.Linear(2048, 7)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)
        return x1, x2, x3

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

model = ResNet34().to(device)
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


# Validation
best_recall = -1
for epoch in range(1, EPOCHS+1):
    print('Epoch {}/{} '.format(epoch, EPOCHS))
    train(model, trn_loader, optimizer, epoch)
    val_loss, val_recall = evaluate(model, vid_loader)
    scheduler.step(val_loss)
    if val_recall > best_recall:
            if not os.path.isdir("checkpoint"):
                os.makedirs("checkpoint")

            print('###### Model Save, Validation Recall is : {:.4f}, Loss is : {:.4f}'.format(val_recall, val_loss))
            torch.save(model.state_dict(), './checkpoint/augmix_resnet34_saved_weights.pth')
            best_recall = val_recall

print('Time to train model: {} mins'.format(round((time.time() - start_time) / 60, 2)))