import os
import argparse

import pytorch_lightning as pl
import torch_optimizer
from torch import nn, optim

from src.dataset import CelebDF, FaceForensics, GANDataset
from src.dataset import augmentations as aug


def config_optimizers(params, args):
    optimizer = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(params, lr=config["lr"])
    # elif args.optimizer == 'ranger':
    #     optimizer = torch_optimizer.Ranger(params,
    #                             lr=args.learning_rate,
    #                             weight_decay=args.weight_decay)
    return optimizer


def config_schedulers(optimizer, args):
    scheduler = None
    if args.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
            verbose=True,
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, verbose=True
        )
    elif args.scheduler == "cosinewarm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=args.tmult, eta_min=0.0001, verbose=True
        )

    return scheduler


def config_transforms(
    mode=None, type=None, input_size=None, crop_size=None, validation=False
):
    if mode == "gan":
        transforms = (
            aug.get_gan_validation_augmentations(
                resize_size=input_size, crop_size=crop_size
            )
            if validation
            else aug.get_gan_training_augmentations(
                aug_type=type, resize_size=input_size, crop_size=crop_size
            )
        )
    elif mode == "df":
        transforms = (
            aug.get_df_validation_augmentations(input_size=input_size)
            if validation
            else aug.get_df_training_augmentations(df_aug=type, input_size=input_size)
        )
    else:
        return ValueError("aug type not implemented")
    return transforms


def config_datasets(**kwargs):
    """
    return pl datamodule that you can use to get dataloaders
    """
    # convert kwargs to namespace
    kwargs = argparse.Namespace(**kwargs)
    assert os.path.exists(kwargs.dataset_path), "DATASET DOES NOT EXIST"
    if kwargs.dataset == "ff":
        dm = FaceForensics.FaceForensics(
            dataset_path=kwargs.dataset_path,
            batch_size=kwargs.batch_size,
            num_workers=kwargs.num_workers,
            train_transforms=kwargs.train_transforms,
            validation_transforms=kwargs.validation_transforms,
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            video_level=kwargs.video_level,
            balance=True,
            pin_memory=kwargs.pin_memory,
            distributed=kwargs.distributed,
            rank=kwargs.rank,
        )
    elif kwargs.dataset == "celebdf":
        dm = CelebDF.CelebDF(
            dataset_path=kwargs.dataset_path,
            batch_size=kwargs.batch_size,
            num_workers=kwargs.num_workers,
            train_transforms=kwargs.train_transforms,
            validation_transforms=kwargs.validation_transforms,
            csv_names=["train_index.csv", "val_100.csv", "test_index.csv"],
            video_level=kwargs.video_level,
            pin_memory=kwargs.pin_memory,
            distributed=kwargs.distributed,
            rank=kwargs.rank,
        )
    elif kwargs.dataset == "gandataset":
        dm = GANDataset.GANDataset(
            datasets_path=kwargs.dataset_path,
            csv_paths=kwargs.csv_paths,
            batch_size=kwargs.batch_size,
            num_workers=kwargs.num_workers,
            train_transforms=kwargs.train_transforms,
            validation_transforms=kwargs.validation_transforms,
            pin_memory=kwargs.pin_memory,
            distributed=kwargs.distributed,
            rank=kwargs.rank,
        )
    else:
        return ValueError("DATASET NAME NOT FOUND")
    print("DM defined")
    pl.seed_everything(1)
    dm.prepare_data()
    return dm
