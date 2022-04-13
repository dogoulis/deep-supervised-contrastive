import json
import os
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from src.dataset.utils import get_batch_sampler
from torch.utils.data import DataLoader, Dataset


class CelebDF(pl.LightningDataModule):
    """
    PL DataModule to load CelebDF dataset.
    Takes into account sampling and balancing.
    """

    def __init__(
        self,
        dataset_path,
        batch_size=32,
        num_workers=48,
        train_transforms=None,
        validation_transforms=None,
        target_transforms=None,
        csv_names=None,
        sampling=1,
        balance=None,
        video_level=False,
    ):
        """
        Parameters:
        -----------
        dataset_path : str
            The path to the dataset. Assumes that quality has been
            selected in the path, i.e.  /path/to/dataset/quality/
        batch_size : int, optional
            the size of the batches that the dataloaders output
            (default is 32)
        num_workers : int, optional
            the number of workers for the DataLoaders (default is 48)
        transforms : Compose object from Albumentations, optional
            transforms to be applied to the images before returning them
        target_transforms : Compose object from Albumentations, optional
            transforms to be applied to the targets or labels before
            returning them
        csv_names : (str, str, str), Optional
            Is a tuple of 3 str of the csv names for train, val, test.
            The naming scheme is: {set}_{sampling*100}_{b if balance is True}.csv
            e.g. train_10_b.csv . The image paths in the csvs are relative
            to the basepath. If csv_names is not provided then sample and balance (if true)
            the dataset.
        sampling : str, float, optional
            If sample=='one', one frame is randomly chosen from each selected
            video. If sample is a float then it represents the sampling ratio
            and it should be in the range (0,1). If sample==None all the frames
            are returned.
        balance : bool, optional
            whether to balance the dataset between the two classes. Balancing
            takes place at video level (default is False)
        video_level : bool, optional
            whether to return the video-level id or frame path of each frame.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.validation_transforms = validation_transforms
        self.target_transforms = target_transforms
        self.csv_names = csv_names
        self.train_csv = self.val_csv = self.test_csv = None
        if csv_names is not None:
            self.train_csv, self.val_csv, self.test_csv = csv_names
        self.sampling = sampling
        self.balance = balance
        self.video_level = video_level

    def prepare_data(self):
        # this gets called one time by only one gpu
        # DO NOT assign state in here
        # if we dont have ready csvs and we have a sampling ration
        # then sample and store the csvs
        if self.csv_names is not None:
            train_csv, val_csv, test_csv = self.csv_names
            assert os.path.exists(
                os.path.join(self.dataset_path, train_csv)
            ), "train_csv provided is not a valid path"
            assert os.path.exists(
                os.path.join(self.dataset_path, val_csv)
            ), "val_csv provided is not a valid path"
            assert os.path.exists(
                os.path.join(self.dataset_path, test_csv)
            ), "test_csv provided is not a valid path"
            print("CSV NAMES ACCEPTED, SKIPPING PREPARE_DATA ...")
            return

        if self.csv_names is None and self.sampling is not None:
            print("\tCSVs NOT PROVIDED")
            # assuming that if there are already csvs the user would have
            # provided them as in csv_names,
            # sampling
            postfix = f"_{int(self.sampling*100)}{'_b_' if self.balance else ''}.csv"

            # train data
            train_data = pd.read_csv(os.path.join(self.dataset_path, "train_index.csv"))
            if self.sampling != 1:
                print("\tSAMPLING TRAIN DATA...")
                train_data = (
                    train_data.groupby("label")
                    .sample(frac=self.sampling, replace=False)
                    .sample(frac=1)
                    .reset_index(drop=True)
                )
            if self.balance:
                print("\tBALANCING TRAIN DATA...")
                real = train_data[train_data.label == 0]
                fake = train_data[train_data.label == 1]
                m = min(len(real), len(fake))
                train_data = (
                    pd.concat([real[:m], fake[:m]])
                    .sample(frac=1)
                    .reset_index(drop=True)
                )

            # test data
            test_data = pd.read_csv(os.path.join(self.dataset_path, "test_index.csv"))
            if self.sampling != 1:
                print("\tSAMPLING TEST DATA...")
                test_data = (
                    test_data.groupby("label")
                    .sample(frac=self.sampling, replace=False)
                    .sample(frac=1)
                    .reset_index(drop=True)
                )
            if self.balance:
                print("\tBALANCING TEST DATA...")
                real = test_data[test_data.label == 0]
                fake = test_data[test_data.label == 1]
                m = min(len(real), len(fake))
                test_data = (
                    pd.concat([real[:m], fake[:m]])
                    .sample(frac=1)
                    .reset_index(drop=True)
                )

            # train val split
            val_len = int(np.floor(len(train_data) * 0.2))
            val_data, train_data = train_data[:val_len], train_data[val_len:]

            train_data.to_csv(
                os.path.join(self.dataset_path, "train" + postfix), index=False
            )
            val_data.to_csv(
                os.path.join(self.dataset_path, "val" + postfix), index=False
            )
            test_data.to_csv(
                os.path.join(self.dataset_path, "test" + postfix), index=False
            )
            print("\tDATA PREPARATION COMPLETE")

    def setup(self, stage=None):
        # steps that should be done on every gpu
        # like splitting data, applying transfroms
        if self.csv_names is None:
            postfix = f"_{int(self.sampling*100)}{'_b_' if self.balance else ''}.csv"
            self.train_csv = "train" + postfix
            self.val_csv = "val" + postfix
            self.test_csv = "test" + postfix
        if stage in (None, "fit"):
            train_df = pd.read_csv(os.path.join(self.dataset_path, self.train_csv))
            # relative to absolute image paths, based on the provided dataset path
            train_df.path = train_df.path.apply(
                lambda x: os.path.join(self.dataset_path, x)
            )
            self.train_dataset = CelebDFDataset(
                train_df.path,
                train_df.label,
                self.train_transforms,
                self.target_transformsm,
                self.video_level,
            )
            val_df = pd.read_csv(os.path.join(self.dataset_path, self.val_csv))
            val_df.path = val_df.path.apply(
                lambda x: os.path.join(self.dataset_path, x)
            )
            self.val_dataset = CelebDFDataset(
                val_df.path,
                val_df.label,
                self.validation_transforms,
                self.target_transforms,
                self.video_level,
            )
        if stage in (None, "test"):
            print("Reading CSV...")
            test_df = pd.read_csv(os.path.join(self.dataset_path, self.test_csv))
            test_df.path = test_df.path.apply(
                lambda x: os.path.join(self.dataset_path, x)
            )
            print("Done reading CSV...")
            self.test_dataset = CelebDFDataset(
                test_df.path,
                test_df.label,
                self.validation_transforms,
                self.target_transforms,
                self.video_level,
            )

    def train_dataloader(self):
        # return train loader
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


class CelebDFDataset(Dataset):
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

        if self.transforms is None:
            self.transforms = ToTensorV2()
        image = self.transforms(image=img)["image"]

        label = self.labels[idx]
        if self.target_transforms is None:
            self.target_transforms = lambda x : torch.tensor(x).float()
        label = self.target_transforms(label)

        id = self.imgs[idx]
        return image, id, label
