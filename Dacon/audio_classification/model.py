import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class audio_resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.org_model = torchvision.models.resnet34()
        self.org_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.features = nn.Sequential(*(list(self.org_model.children())[:-1]))
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = self.features(x)
        bs,c,w,h = x.size()
        x = x.view(bs, -1)
        out = self.fc(x)
        return out
