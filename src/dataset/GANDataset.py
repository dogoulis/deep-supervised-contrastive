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
        datasets_path,
        csv_paths,
        batch_size=32,
        num_workers=8,
        train_transforms=None,
        validation_transforms=None,
    ):
        super().__init__()
        self.datasets_path = datasets_path
        self.csv_paths = csv_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.validation_transforms = validation_transforms

    def prepare_data(self):
        # this gets called one time by only one gpu
        # DO NOT assign state in here
        pass

    def setup(self, stage=None):
        # steps that should be done on every gpu
        # like splitting data, applying transfroms
        if stage in (None, "fit"):
            self.train_dataset = dataset2(
                datasets_path=self.datasets_path,
                csv_path=self.csv_paths[0],
                transforms=self.train_transforms,
            )
            self.val_dataset = dataset2(
                datasets_path=self.datasets_path,
                csv_path=self.csv_paths[1],
                transforms=self.validation_transforms,
            )
        if stage in (None, "test"):
            self.test_dataset = dataset2(
                datasets_path=self.datasets_path,
                csv_path=self.csv_paths[2],
                transforms=self.validation_transforms,
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
            shuffle=False,
            num_workers=self.num_workers,
        )


class dataset2(Dataset):
    def __init__(self, datasets_path, csv_path, transforms):
        self.datasets_path = datasets_path
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)
        self.imgs = self.dataset.image_path.values
        self.labels = self.dataset.label.values
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(os.path.join(self.datasets_path, self.imgs[idx]))
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
        id = self.imgs[idx]
        return image, id, label