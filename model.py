import zipfile

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
"""
class ImgDataset(Dataset):
    def __init__(self, file, transform=train_transform):
        data = np.load(file, allow_pickle=True)
       # self.x = torch.Tensor([i[0] for i in data]).view(-1, 128, 128)
        self.x = torch.Tensor([i[0] for i in data]).view(-1, 28, 28)
        # Scaling the features
        self.x = self.x / 255.0
        # Getting the target
        y = torch.Tensor([i[1] for i in data])
        self.y = []
        real = torch.argmax(y)
        for i in range(len(y)):
            real = torch.argmax(y[i])
            self.y.append(real.item())

        print(len(self.y))
        count = 0
        for i in range(len(self.y)):
            if self.y[i] == 0:
                count += 1
        print(count)
        self.y = torch.Tensor(self.y)

        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
"""
#TODO: FUNCTION THAT SCANS THE DATASET AND CONVERT IT INTO IMAGES AND STORE THEM INTO THEIR RESPECTIVE FOLDER

class CatsVSDogs():
    # Size of the image
   # IMG_SIZE = 128
    IMG_SIZE = 28

    # Directory location
    #TODO: CHANGE IT
    TSHIRT = 'datasets/TODO'
    TROUSER = 'datasets/TODO'
    PULLOVER = 'datasets/TODO'
    DRESS = 'datasets/TODO'
    COAT = 'datasets/TODO'
    SANDAL = 'datasets/TODO'
    SHIRT = 'datasets/TODO'
    SNEAKER = 'datasets/TODO'
    BAG = 'datasets/TODO'
    ANKLEBOOT = 'datasets/TODO'

    # Labels for cats and dogs
    LABELS = {TSHIRT: 0, TROUSER: 1, PULLOVER: 2, DRESS: 3, COAT: 4, SANDAL: 5, SHIRT: 6, SNEAKER: 7, BAG: 8, ANKLEBOOT: 9}

    # Initializing variables
    all_data = []
    train_data = []
    test_data = []
    tshirtcount = 0
    trousercount = 0
    pullovercount = 0
    dresscount = 0
    coatcount = 0
    sandalcount = 0
    shirtcount = 0
    sneakercount = 0
    bagcount = 0
    anklebootcount = 0

    def make_training_data(self):
        for label in self.LABELS:

            # Looping through each pictures
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)

                    # Reading images and converting to grayscale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                    # Resizing images
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    # Getting the training data
                    self.all_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # Checking distribution of data
            #        if label == self.CATS:
           #             self.catcount += 1
             #       elif label == self.DOGS:
              #          self.dogcount += 1
                    if label == self.TSHIRT:
                        self.tshirtcount += 1
                    elif label == self.TROUSER:
                        self.trousercount += 1
                    elif label == self.PULLOVER:
                        self.pullovercount += 1
                    elif label == self.DRESS:
                        self.dresscount += 1
                    elif label == self.COAT:
                        self.coatcount += 1
                    elif label == self.SANDAL:
                        self.sandalcount += 1
                    elif label == self.SHIRT:
                        self.shirtcount += 1
                    elif label == self.SNEAKER:
                        self.sneakercount += 1
                    elif label == self.BAG:
                        self.bagcount += 1
                    elif label == self.ANKLEBOOT:
                        self.anklebootcount += 1
                        
                except Exception as e:
                    pass

            np.random.shuffle(self.all_data)
            val = int(len(self.all_data) * 0.3)
            self.train_data = self.all_data[:-val]
            self.test_data = self.all_data[-val:]

            np.save("train_data_28.npy", self.train_data)
            np.save("test_data_28.npy", self.test_data)
            print("T-shirts/Tops: ", self.tshirtcount)
            print("Trousers: ", self.trousercount)
            print("Pullovers: ", self.pullovercount)
            print("Dresses: ", self.dresscount)
            print("Coats: ", self.coatcount)
            print("Sandals: ", self.sandalcount)
            print("Shirts: ", self.shirtcount)
            print("Sneakers: ", self.sneakercount)
            print("Bags: ", self.bagcount)
            print("Ankle Boots: ", self.anklebootcount)

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

         #   nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
         #   nn.BatchNorm2d(256),
         #   nn.ReLU(),
         #   nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

         #   nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
         #   nn.BatchNorm2d(512),
         #   nn.ReLU(),
         #   nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

         #   nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
         #   nn.BatchNorm2d(512),
         #   nn.ReLU(),
         #   nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
         #   nn.Linear(128, 2),
            nn.Softmax(dim=10)
        )

    def forward(self, x):
        print(x.shape)
        out = self.cnn(x)
        print("test1")
        out = out.view(out.size()[0], -1)
        return self.fc(out)
