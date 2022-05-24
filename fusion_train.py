import argparse
from ast import arg
import os
from random import shuffle, triangular
from glob import glob


import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import device, nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import random
from src.pytorch_balanced_sampler.sampler import SamplerFactory

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
    wandb.run.name = wandb.run.name + '_' + args.model + '_' + '_'.join(args.loss)
    wandb.define_metric("train-epoch-loss", summary="min")
    wandb.define_metric("train-steploss", summary="min")
    wandb.define_metric('validation-loss', summary="min")
    wandb.define_metric('validation-accuracy', summary="max")

    # model definition
    model = Model(config=vars(args)).to(args.device)

    # define training transforms/augmentations
    gan_train_transforms = config_transforms(
        mode='gan',
        validation=False,
        type='geometric',
        input_size=224,
        crop_size=224,
    )

    gan_validation_transforms = config_transforms(
        mode='gan',
        validation=True,
        input_size=224,
        crop_size=224,
    )

    df_train_transforms = config_transforms(
        mode='df',
        validation=False,
        type='rand',
        input_size=224,
        crop_size=None,
    )

    df_validation_transforms = config_transforms(
        mode='df',
        validation=True,
        input_size=224,
        crop_size=None,
    )

    gandm = config_datasets(
        dataset='gandataset',
        dataset_path='/fssd1/user-data/dogoulis/',
        csv_paths=[
            '/fssd1/user-data/dogoulis/ffhq_st2_train.csv',
            '/fssd1/user-data/dogoulis/ffhq_st2_valid.csv',
            '/fssd1/user-data/dogoulis/ffhq_st2_test.csv',
        ],
        batch_size=32,
        num_workers=8,
        train_transforms=gan_train_transforms,
        validation_transforms=gan_validation_transforms,
        video_level=False,
        balance=False
    )
    gandm.train_dataset.labels = np.where(gandm.train_dataset.labels == 1, 2, 0)
    gandm.val_dataset.labels = np.where(gandm.val_dataset.labels == 1, 2, 0)
    gandm.test_dataset.labels = np.where(gandm.test_dataset.labels == 1, 2, 0)

    dfdm = config_datasets(
        dataset='ff',
        dataset_path='/fssd1/user-data/spirosbax/data/FaceForensics++/',
        csv_paths=None,
        batch_size=32,
        num_workers=8,
        manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        train_transforms=df_train_transforms,
        validation_transforms=df_validation_transforms,
        video_level=False,
        balance=False
    )

    train_dataset = ConcatDataset([gandm.train_dataset, dfdm.train_dataset])
    validation_dataset = ConcatDataset([gandm.val_dataset, dfdm.val_dataset])
    test_dataset = ConcatDataset([gandm.test_dataset, dfdm.test_dataset])

    # define training sampler
    train_class_idxs = [
        [i + len(gandm.train_dataset) for i, l in enumerate(dfdm.train_dataset.labels) if l == 0],
        [i + len(gandm.train_dataset) for i, l in enumerate(dfdm.train_dataset.labels) if l == 1],
        [i for i, l in enumerate(gandm.train_dataset.labels) if l == 2]
    ]
    random.shuffle(train_class_idxs[0])
    random.shuffle(train_class_idxs[1])
    random.shuffle(train_class_idxs[2])

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_sampler=SamplerFactory().get(
            class_idxs=train_class_idxs,
            batch_size=args.batch_size,
            n_batches=len(train_dataset) // args.batch_size,
            alpha=1,
            kind="fixed",
        )
    )

    # define validation sampler
    validation_class_idxs = [
        [i + len(gandm.val_dataset) for i, l in enumerate(dfdm.val_dataset.labels) if l == 0],
        [i + len(gandm.val_dataset) for i, l in enumerate(dfdm.val_dataset.labels) if l == 1],
        [i for i, l in enumerate(gandm.val_dataset.labels) if l == 2]
    ]
    random.shuffle(validation_class_idxs[0])
    random.shuffle(validation_class_idxs[1])
    random.shuffle(validation_class_idxs[2])

    val_dataloader = DataLoader(
        validation_dataset,
        num_workers=8,
        batch_sampler=SamplerFactory().get(
            class_idxs=validation_class_idxs,
            batch_size=args.batch_size,
            n_batches=len(validation_dataset) // args.batch_size,
            alpha=1,
            kind="fixed",
        )
    )

    # define test sampler
    test_class_idxs = [
        [i + len(gandm.test_dataset) for i, l in enumerate(dfdm.test_dataset.labels) if l == 0],
        [i + len(gandm.test_dataset) for i, l in enumerate(dfdm.test_dataset.labels) if l == 1],
        [i for i, l in enumerate(gandm.test_dataset.labels) if l == 2]
    ]
    random.shuffle(test_class_idxs[0])
    random.shuffle(test_class_idxs[1])
    random.shuffle(test_class_idxs[2])

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=8,
        batch_sampler=SamplerFactory().get(
            class_idxs=test_class_idxs,
            batch_size=args.batch_size,
            n_batches=len(test_dataset) // args.batch_size,
            alpha=1,
            kind="fixed",
        )
    )

    # define optimizer and scheduler
    optimizer = config_optimizers(model.parameters(), args)
    scheduler = config_schedulers(optimizer, args)

    # define the criterion:
    criterion = nn.CrossEntropyLoss()

    # set up fp16
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # checkpointing - directories
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
            model, dataloader=val_dataloader, args=args, criterion=criterion
        )

        # TODO add more functionality here
        # e.g. better names for checkpoints and patience for early stopping
        if val_results["val_loss"] < min_loss:
            min_loss = val_results["val_loss"].copy()
            ckpt_name = f"{wandb.run.name}_epoch_{epoch}_val_loss_{val_results['val_loss']:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(args.save_model_path, ckpt_name))
    
    # get best checkpoint
    print("Loading best checkpoint...")
    saved_ckpts = glob(os.path.join(args.save_model_path, wandb.run.name + '*.pt'))
    saved_ckpts_epochs = [int(x.split('/')[-1].split('_')[-4]) for x in saved_ckpts]
    best_idx = saved_ckpts_epochs.index(max(saved_ckpts_epochs))
    best_ckpt = saved_ckpts[best_idx]

    # load best checkpoint
    del model
    model = Model(config=vars(args)).to(args.device)
    model.load_state_dict(torch.load(best_ckpt))

    # test on test data and log results
    test_results = validate_epoch(
        model, dataloader=test_dataloader, args=args, criterion=criterion, testing=True
    )


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
        y = y.type(torch.LongTensor)
        y = y.to(args.device)

        # select the real and fake indexes at batches
        real_idxs = y == 0
        fake_idxs = y == 1
        synthetic_idxs = y == 2

        real_class_batch = x[real_idxs]
        fake_class_batch = x[fake_idxs]
        synthetic_class_batch = x[synthetic_idxs]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # pass the real and fake batches through the backbone network and then through the projectors
            z_real = model.real_projector(model(real_class_batch))
            z_fake = model.fake_projector(model(fake_class_batch))
            z_synthetic = model.synthetic_projector(model(synthetic_class_batch))

            # pass the batch through the classifier head
            output = model.head(model(x))
            output = torch.softmax(output, dim=1)

            # mixed loss calculation
            # get the log of barlow losses
            loss = 0
            if 'bce' in args.loss:
                loss += criterion(output, y)
            if 'barlow' in args.loss:
                loss += BarlowTwinsLoss(z_real).log() + BarlowTwinsLoss(z_fake).log() + BarlowTwinsLoss(z_synthetic).log()
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
def validate_epoch(model, dataloader, args, criterion, testing=False):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, id, y in dataloader:
        x = x.to(args.device)
        y = y.type(torch.LongTensor)
        y = y.to(args.device)

        output = model.head(model(x))
        output = torch.softmax(output, dim=1)
        loss = criterion(output, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        output = output.argmax(dim=1)
        y_true.append(y.cpu())
        y_pred.append(output.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    print(y_true)
    y_pred = torch.cat(y_pred, 0).numpy()
    print(y_pred)
    tot_loss = np.mean(running_loss)
    wandb.log({"validation-loss": tot_loss}) if not testing else wandb.log({"test-loss": tot_loss})
    acc = 100.0 * np.mean(y_true == y_pred)
    wandb.log({"validation-accuracy": acc}) if not testing else wandb.log({"test-accuracy": acc})
    return {"val_acc": acc, "val_loss": tot_loss} if not testing else {"test_acc": acc, "test_loss": tot_loss}


if __name__ == "__main__":
    main()
