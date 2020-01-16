import torch
from torchvision.models import resnet50
from models.LocallyConnected2d import LocallyConnected2d
from PIL import Image
import numpy as np


class FeatureMapping(torch.nn.Module):
    def __init__(self, feat_size=[18,512]):
        super().__init__()

        self.feat_size = feat_size
        self.activation = torch.nn.ELU()
        self.dense= torch.nn.Linear((18*512), (18*512))
    def forward(self, input_feats):
        out=input_feats.view((-1,18*512))
        for _ in range(4):
            out=self.dense(out)
            out=torch.relu(out)
        return out