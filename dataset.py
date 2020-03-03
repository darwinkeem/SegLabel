import os
import pandas as pd
import numpy as np

from PIL import Image
from cv2 import cv2

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils


class SegDataset(Dataset):
    def __init__(self, root='./data/', split='train', transforms=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transforms

        self.file = []

        if self.split == 'train':
            textfile = open('./data/split/train_file.txt', 'r')
            read = textfile.read()
            self.file = read.split('\n')
            self.file.remove('')
        elif self.split == 'val':
            textfile = open('./data/split/val_file.txt', 'r')
            read = textfile.read()
            self.file = read.split('\n')
            self.file.remove('')
        elif self.split == 'test':
            textfile = open('./data/split/test_file.txt', 'r')
            read = textfile.read()
            self.file = read.split('\n')
            self.file.remove('')

    def __getitem__(self, idx):
        img_loc, mask_loc = self.file[idx].split(' ')[0], self.file[idx].split(' ')[1]

        img = cv2.imread(img_loc)
        mask = cv2.imread(mask_loc)

        sample = (img, mask)

        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    ds = SegDataset()
    cv2.imshow('asadf', ds.__getitem__(10)[0])
    cv2.waitKey()
    cv2.imshow('asadf', ds.__getitem__(10)[1])
    cv2.waitKey()
            