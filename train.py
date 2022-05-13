import argparse
from ast import arg
import os
from random import shuffle, triangular


import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import device, nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.losses import BarlowTwinsLoss, supcon
from src.model import Model
from src.argparser_args import get_argparser
from src.configurators import (
    config_optimizers,
    config_schedulers,
    config_transforms,
    config_datasets,
)

# CMD ARGUMENTS
parser = get_argparser()
args = parser.parse_args()
print(args)


def main():
    # initialize weights and biases
    wandb.init(
        entity=args.entity,
        project=args.project_name,
        name=args.name,
        config=vars(args),
        group=args.run_group,
        save_code=True,
    )
    args.device = torch.device(args.device)

    # model definition
    model = Model(config=vars(args)).to(args.device)

    # define training transforms/augmentations
    train_transforms = config_transforms(
        mode=args.augmentations_mode,
        validation=False,
        type=args.augmentations_type,
        input_size=args.input_size,
        crop_size=args.crop_size,
    )

    validation_transforms = config_transforms(
        mode=args.augmentations_mode,
        validation=True,
        input_size=args.input_size,
        crop_size=args.crop_size,
    )

    dm = config_datasets(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        csv_paths=[args.train_path, args.validation_path, args.test_path],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        train_transforms=train_transforms,
        validation_transforms=validation_transforms,
        video_level=False,
        balance=False
    )
    if isinstance(dm, pl.LightningDataModule):
        dm.prepare_data()
        dm.setup(stage="fit")
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    # define optimizer and scheduler
    optimizer = config_optimizers(model.parameters(), args)
    scheduler = config_schedulers(optimizer, args)

    # define the criterion:
    criterion = nn.BCEWithLogitsLoss()

    # set up fp16
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # checkpointing - directories
    save_model_dir = args.save_model_path
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.save_backbone_path):
        os.makedirs(args.save_backbone_path)

    # define value for min-loss
    min_loss = float("inf")

    print("Training starts...")
    for epoch in range(args.epochs):
        wandb.log({"epoch": epoch})
        train_epoch(
            model,
            train_dataloader=train_dataloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            fp16_scaler=fp16_scaler,
            epoch=epoch,
        )
        val_results = validate_epoch(
            model, val_dataloader=val_dataloader, args=args, criterion=criterion
        )

        # TODO add more functionality here
        # e.g. better names for checkpoints and patience for early stopping
        if val_results["val_loss"] < min_loss:
            min_loss = val_results["val_loss"].copy()
            ckpt_name = f"{wandb.run.name}_epoch_{epoch}_val_loss_{val_results['val_loss']:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(save_model_dir, ckpt_name))


def train_epoch(
    model,
    train_dataloader,
    args,
    optimizer,
    criterion,
    scheduler=None,
    fp16_scaler=None,
    epoch=0,
):
    # to train only the classification layer
    model.train()
    epoch += 1
    running_loss = []
    pbar = tqdm(train_dataloader, desc=f"epoch {epoch}.", unit="iter")

    for batch, (x, id, y) in enumerate(pbar):
        x = x.to(args.device)
        y = y.to(args.device)

        # select the real and fake indexes at batches
        real_idxs = y == 0
        fake_idxs = y == 1

        real_class_batch = x[real_idxs]
        fake_class_batch = x[fake_idxs]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # pass the real and fake batches through the backbone network and then through the projectors
            z_real = model.real_projector(model(real_class_batch))
            z_fake = model.fake_projector(model(fake_class_batch))

            # pass the batch through the classifier head
            output = model.head(model(x)).flatten()

            # mixed loss calculation
            # get the log of barlow losses
            loss = 0
            if 'bce' in args.loss:
                loss += criterion(output, y)
            if 'barlow' in args.loss:
                loss += BarlowTwinsLoss(z_real).log() + BarlowTwinsLoss(z_fake).log()
            if 'supcon' in args.loss:
                loss += supcon(torch.cat((z_fake, z_real), axis=0), torch.cat((y[fake_idxs], y[real_idxs]), axis=0))


        # mixed-precesion if given in arguments
        if fp16_scaler:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss.append(loss.detach().cpu().numpy())
        # log mean loss for the last 10 batches:
        if (batch + 1) % 10 == 0:
            wandb.log({"train-steploss": np.mean(running_loss[-10:])})

    # scheduler
    if scheduler is not None:
        scheduler.step()
        wandb.log({'learning_rate': scheduler.get_lr()[0]})

    train_loss = np.mean(running_loss)
    wandb.log({"train-epoch-loss": train_loss})

    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, id, y in val_dataloader:
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        outputs = model.head(model(x))
        loss = criterion(outputs, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)
        y_true.append(y.cpu())
        y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    y_pred = torch.cat(y_pred, 0).numpy()
    val_loss = np.mean(running_loss)
    wandb.log({"validation-loss": val_loss})
    acc = 100.0 * np.mean(y_true == y_pred)
    wandb.log({"validation-accuracy": acc})
    return {"val_acc": acc, "val_loss": val_loss}


if __name__ == "__main__":
    main()
