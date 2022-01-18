import numpy as np
import torch
from kaggle import KaggleApi
from torchvision import datasets
from torch import nn
from torch.utils.data import Dataset
import cv2
import os

from torchvision.transforms import transforms
from tqdm import tqdm

#load datasets train and test different
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class CatsVSDogs():
    def train(self, trained):
        dataset = datasets.FashionMNIST(root="./data", train=trained, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return dataset
# CNN
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input [1, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # [64, 28, 28]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 14, 14]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # [128, 7, 7]
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
