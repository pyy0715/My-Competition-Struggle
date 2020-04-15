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

from utils import macro_recall, macro_recall_multi, calc_macro_recall

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
BATCH_SIZE = 1
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
    def __init__(self, csv, img_height, img_width):
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(f'./input/test_images/{img_id}.pkl')
        img = img.reshape(self.img_height, self.img_width)
        img = 255 - img
        img = crop_resize(img)
        img = img[:, :, np.newaxis]

        return torch.tensor(img, dtype=torch.float).permute(2, 0, 1)

df_test = pd.read_csv('./input/test.csv')
df_test.drop(['row_id','component'], axis=1, inplace=True)
df_test.drop_duplicates(inplace=True)

test_dataset = BengaliDataset(csv=df_test, img_height=HEIGHT, img_width=WIDTH)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)


        # grapheme_root
        self.fc1 = nn.Linear(1024, 168)
        # vowel_diacritic
        self.fc2 = nn.Linear(1024, 11)
        # consonant_diacritic
        self.fc3 = nn.Linear(1024, 7)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)
        return x1, x2, x3

model = ResNet().to(device)
model.load_state_dict(torch.load('./checkpoint/resnet_saved_weights.pth'))


model.eval()
predictions = []
with torch.no_grad():
        for idx, (img) in enumerate(test_loader):
            img = img.to(device)
            pred_grapheme, pred_vowel, pred_consonan = model(img)
            predictions.append(pred_grapheme.argmax(1).cpu().detach().numpy())
            predictions.append(pred_vowel.argmax(1).cpu().detach().numpy())
            predictions.append(pred_consonan.argmax(1).cpu().detach().numpy())

print(predictions)
submission = pd.read_csv('./input/sample_submission.csv')
submission.target = np.hstack(predictions)
submission.to_csv('./pred/submission.csv',index=False)