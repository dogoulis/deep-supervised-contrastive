import os

import cv2
import numpy as np
import torch
import torchdata.datapipes as dp
from albumentations.pytorch import ToTensorV2
from src.dataset.utils import get_batch_sampler
from torch.utils.data import DataLoader, Dataset


class OpenForensics:
    def __init__(
        self,
        dataset_path,
        names,
        batch_size=32,
        num_workers=8,
        train_transforms=None,
        validation_transforms=None,
    ):
        self.dataset_path = dataset_path
        self.names = names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.validation_transforms = validation_transforms

        assert len(names) == 3, "NEED 3 FOLDER NAMES FOR TRAIN, VAL, TEST"
        train_name, val_name, test_name = names
        for i, name in enumerate(names):
            path = os.path.join(dataset_path, name)
            assert os.path.exists(path)
            assert os.path.exists(os.path.join(path, "real"))
            assert os.path.exists(os.path.join(path, "fake"))

            real = list(dp.iter.FileLister(os.path.join(path, "real")))
            fake = list(dp.iter.FileLister(os.path.join(path, "fake")))
            labels = [0] * len(real) + [1] * len(fake)
            if i == 0:
                self.train_dataset = OpenForensicsDataset(
                    real + fake, labels, transforms=train_transforms
                )
            elif i == 1:
                self.val_dataset = OpenForensicsDataset(
                    real + fake, labels, transforms=self.validation_transforms
                )
            elif i == 2:
                self.test_datset = OpenForensicsDataset(
                    real + fake, labels, transforms=self.validation_transforms
                )

    def train_dataloader(self):
        # return val loader
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_sampler=get_batch_sampler(
                dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
            ),
        )

    def val_dataloader(self):
        # return val loader
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_sampler=get_batch_sampler(
                dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True
            ),
        )

    def test_dataloader(self):
        # return test loader
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_sampler=get_batch_sampler(
                dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
            ),
        )


class OpenForensicsDataset(Dataset):
    def __init__(
        self, imgs, labels, transforms=None, target_transforms=None, video_level=False
    ):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.video_level = video_level

    def __len__(self):
        return len(self.imgs)

    def get_video_id(self, x):
        # get video id from path
        return "/".join(x.split("/")[:-3])

    def get_comp_id(self, x):
        # get comp id from path
        return "_".join(x.split("_")[:-1])

    def get_vid_from_comp_id(self, x):
        # get vid id from comp
        return "/".join(x.split("/")[:-3])

    def __getitem__(self, idx):
        # imread returns 0-255 HWC BGR numpy array
        # albumentation needs 0-255 HWC !RGB! numpy array
        # try to load img 10 times
        img = None
        for i in range(10):
            try:
                img = cv2.imread(self.imgs[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                continue
            else:
                break
        else:
            print(f"COULD NOT LOAD IMG: {self.imgs[idx]}")
        label = self.labels[idx]

        if self.transforms is None:
            self.transforms = ToTensorV2()
        image = self.transforms(image=img)["image"]

        if self.target_transforms:
            label = self.target_transforms(label)
        else:
            label = torch.tensor(label).float()
        id = self.imgs[idx]
        return image, id, label
