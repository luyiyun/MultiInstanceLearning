import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.models as models
import progressbar as pb


class NormalCnn(nn.Module):

    def __init__(self):
        super(NormalCnn, self).__init__()
        backbones = list(models.resnet50(True).children())
        self.backbone = nn.Sequential(*backbones[:-1])
        self.final = nn.Sequential(
            nn.Linear(backbones[-1].in_features, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze()
        return self.final(x)


def test():
    pass


if __name__ == "__main__":
    test()
