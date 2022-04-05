from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
import os
import pytorch_lightning as pl


class GANDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        batch_size=32,
        num_workers=8,
        transformrs=None,
        root_dir=None,
        csv_names=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transformrs
        self.root_dir = root_dir
        self.csv_names = csv_names  # list of [train_csv, val_csv, test_csv]

    def setup(self, stage=None):

        if stage in (None, "fit"):
            self.train_dataset = dataset2(
                dataset_path=self.csv_names[0],
                root_dir=self.root_dir,
                transforms=self.transforms,
            )
            self.val_dataset = dataset2(
                dataset_path=self.csv_names[1],
                root_dir=self.root_dir,
                transforms=self.transforms,
            )
        if stage in (None, "test"):
            self.test_dataset = dataset2(
                dataset_path=self.csv_names[2],
                root_dir=self.root_dir,
                transforms=self.transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


class dataset2(Dataset):
    def __init__(self, root_dir, dataset_path, transforms):
        self.root_dir = root_dir
        self.dataset = pd.read_csv(dataset_path)
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(os.path.join(self.root_dir, self.imgs[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(self.imgs[idx])
            idx += 1
            image = cv2.imread(os.path.join(self.root_dir, self.imgs[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        if self.transforms:
            tr_img = self.transforms(image=image)
            image = tr_img["image"]
        else:
            image = ToTensorV2(image)
        label = torch.tensor(label).float()

        return image, label
