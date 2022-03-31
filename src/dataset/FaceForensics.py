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
from torch.utils.data import DataLoader, Dataset


class FaceForensics(pl.LightningDataModule):
    """
    DataModule to load FaceForensics++ dataset.
    Takes into account sampling, balancing and specific
    manipulation methods.

    TODO: add quality selection
    """

    def __init__(self,
                 dataset_path,
                 batch_size=32,
                 num_workers=8,
                 transforms=None,
                 target_transforms=None,
                 manipulations=None,
                 sampling=None,
                 balance=False,
                 csv_names=None,
                 video_level=False):
        """
        Parameters
        ----------
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
        manipulations : List of strings, optional
            This list should contain the names of the manipulations from which
            the fake images should be drawn. The names must match exactly the
            folder names. If None, all the manipulations are considered.
        sampling : str, float, optional
            If sample=='one', one frame is randomly chosen from each selected
            video. If sample is a float then it represents the sampling ratio
            and it should be in the range (0,1). If sample==None all the frames
            are returned.
        balance : bool, optional
            whether to balance the dataset between the two classes. Balancing
            takes place at video level (default is False)
        csv_names : list of strings, optional
            List of the csv names to be used for loading the dataset. The names
            are expected to be in the following order: [train, val, test] and
            be placed in the dataset_path directory. If this argument is
            provided then the prepare_data function has no effect.
        video_level : bool, optional
            whether to return the video-level id or frame path of each frame.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.manipulations = manipulations
        self.sampling = sampling
        self.balance = balance
        self.csv_names = csv_names
        self.video_level = video_level

    def prepare_data(self):
        # this gets called one time by only one gpu
        # DO NOT assign state in here
        if self.csv_names is not None:
            train_csv, val_csv, test_csv = self.csv_names
            assert os.path.exists(os.path.join(self.dataset_path, train_csv)), 'train_csv provided is not a valid path'
            assert os.path.exists(os.path.join(self.dataset_path, val_csv)), 'val_csv provided is not a valid path'
            assert os.path.exists(os.path.join(self.dataset_path, test_csv)), 'test_csv provided is not a valid path'
            print('CSV NAMES ACCEPTED, SKIPPING PREPARE_DATA ...')
            return

        real_videos_paths = os.path.join(self.dataset_path,
                                         'original_sequences/'
                                         'c23',
                                         'videos',
                                         '*.mp4')
        real_videos_paths = glob(real_videos_paths)
        if self.manipulations is None:
            fake_videos_paths = os.path.join(self.dataset_path,
                                            'manipulated_sequences/',
                                            '*',
                                            'c23',
                                            'videos',
                                            '*.mp4')
            fake_videos_paths = glob(fake_videos_paths)
        else:
            fake_videos_paths = [item
                                 for manipulation in self.manipulations
                                 for item in glob(os.path.join(
                                     self.dataset_path,
                                     'manipulated_sequences',
                                     manipulation,
                                     'c23',
                                     'videos',
                                     '*.mp4'))]
        # balance dataset
        if self.balance:
            # randomly pick fake videos
            fake_videos_paths = random.sample(fake_videos_paths,
                                              k=len(real_videos_paths))
        # sample videos
        if type(self.sampling) is str and self.sampling == 'one':
            # choose only one frame from each video
            sampled_images = [random.sample(frames, k=1)[0]
                              for video in real_videos_paths+fake_videos_paths
                              for frames in glob(os.path.join(video,'seg*','*.png'))]
            real_sampled_images = sampled_images[:len(real_videos_paths)]
            fake_sampled_images = sampled_images[len(real_videos_paths):]
        elif type(self.sampling) is float:  # sample according to ratio
            real_sampled_images = [
                frame
                for video in real_videos_paths
                for frames in glob(os.path.join(video,'seg*','*.png'))
                for frame in random.sample(frames, k=np.floor(self.sampling*len(frames)))
            ]
            fake_sampled_images = [
                frame
                for video in fake_videos_paths
                for frames in glob(os.path.join(video,'seg*','*.png'))
                for frame in random.sample(frames, k=np.floor(self.sampling*len(frames)))
            ]
        else:  # no sampling, include all frames
            assert self.sampling is None
            real_sampled_images = [
                frame
                for video in real_videos_paths
                for frame in glob(os.path.join(video,'seg*','*.png'))
            ]
            fake_sampled_images = [
                frame
                for video in fake_videos_paths
                for frame in glob(os.path.join(video,'seg*','*.png'))
            ]
        labels = [0] * len(real_sampled_images)
        labels.extend([1] * len(fake_sampled_images))
        images = real_sampled_images + fake_sampled_images

        # get json with video ids
        # load json and get all ids from video set
        all_ids = []
        for video_set in ['train', 'val', 'test']:
            splits_json = os.path.join(self.dataset_path, 'splits', video_set + '.json')
            assert os.path.exists(splits_json)
            with open(splits_json) as e:
                json_array = json.load(e)
            set_ids = [i for x in json_array for i in x]
            all_ids.append(set_ids)
        train_ids, val_ids, test_ids = all_ids

        train_set, val_set, test_set = [], [], []
        for path, label in zip(images, labels):
            vid_name = path.split('/')[-2]
            first_id = vid_name.split('.')[0].split('_')[0]
            if first_id in train_ids:
                train_set.append((path, label))
            elif first_id in val_ids:
                val_set.append((path, label))
            else:
                test_set.append((path, label))

        for name, files in zip(['train', 'val', 'test'],
                               [train_set, val_set, test_set]):
            df = pd.DataFrame(files, columns=['path', 'label'])
            # convert paths to relative
            df.path = df.path.apply(
                lambda x: '/'.join(x.split('/')[-6:]) if 'manipulated' not in x else '/'.join(x.split('/')[-7:])
            )
            df.to_csv(os.path.join(self.dataset_path, f'{name}.csv'))

    def setup(self, stage=None):
        # steps that should be done on every gpu
        # like splitting data, applying transfroms
        if stage in (None, 'fit'):
            train_df = pd.read_csv(os.path.join(
                self.dataset_path,
                self.csv_names[0] if self.csv_names is not None else 'train.csv')
            )
            train_df.path = train_df.path.apply(lambda x: os.path.join(self.dataset_path, x))
            self.train_dataset = FaceForensicsDataset(train_df.path,
                                                      train_df.label,
                                                      self.transforms,
                                                      self.target_transforms,
                                                      self.video_level)
            val_df = pd.read_csv(os.path.join(
                self.dataset_path,
                self.csv_names[1] if self.csv_names is not None else 'val.csv')
            )
            val_df.path = val_df.path.apply(lambda x: os.path.join(self.dataset_path, x))
            self.val_dataset = FaceForensicsDataset(val_df.path,
                                                    val_df.label,
                                                    self.transforms,
                                                    self.target_transforms,
                                                    self.video_level)
        if stage in (None, 'test'):
            test_df = pd.read_csv(os.path.join(
                self.dataset_path,
                self.csv_names[2] if self.csv_names is not None else 'test.csv')
            )
            test_df.path = test_df.path.apply(lambda x: os.path.join(self.dataset_path, x))
            self.test_dataset = FaceForensicsDataset(test_df.path,
                                                     test_df.label,
                                                     self.transforms,
                                                     self.target_transforms,
                                                     self.video_level)

    def train_dataloader(self):
        # return train loader
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # return val loader
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        # return test loader
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


class FaceForensicsDataset(Dataset):

    def __init__(self,
                 imgs,
                 labels,
                 transforms=None,
                 target_transforms=None,
                 video_level=False):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.video_level = video_level

    def __len__(self):
        return len(self.imgs)

    def get_video_id(self, x):
        # get video id from path
        return '/'.join(x.split('/')[:-2])

    def get_comp_id(self, x):
        # get comp id from path
        return '_'.join(x.split('_')[:-1])

    def get_vid_from_comp_id(self, x):
        # get vid id from comp
        return '/'.join(x.split('/')[:-2])

    def __getitem__(self, idx):
        # imread returns 0-255 HWC BGR numpy array
        # albumentation needs 0-255 HWC !RGB! numpy array
        # try to load img 10 times
        img = None
        image = torch.empty(1)
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
        if self.transforms:
            image = self.transforms(image=img)['image']
        if self.target_transforms:
            label = self.target_transforms(label)
        else:
            label = torch.tensor(label).float()
        id = self.imgs[idx]
        return image, id, label