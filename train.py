import argparse
import os
from random import shuffle

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import device, nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch.multiprocessing as mp

from src.losses import BarlowTwinsLoss
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
    args.gpus = torch.cuda.device_count()
    assert args.gpus >= args.world_size, f"gpus: {args.gpus} < world_size: {args.world_size}"

    # print rank, world size and gpu
    print(f"world size: {args.world_size}")
    print(f"gpu: {args.gpus}")

    # initialize weights and biases
    wandb.init(
        entity=args.entity,
        project=args.project_name,
        name=args.name,
        config=vars(args),
        group=args.run_group,
        save_code=True,
    )

    mp.spawn(
        run,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )

def setup(rank, world_size):
    """
    Setup for distributed training
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8890'

    print('os env set')

    # initialize the process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def run(rank, args):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, args.world_size)

    print('setup done')

    args.device = torch.device(rank)
    print(f"Using device: {args.device}")

    # model definition
    model = Model(config=vars(args)).to(args.device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    print('model defined')

    # define training transforms/augmentations
    train_transforms = config_transforms(
        mode=args.augmentations_mode,
        type=args.augmentations_type,
        input_size=args.input_size,
        crop_size=args.crop_size,
        validation=False,
    )

    validation_transforms = config_transforms(
        mode=args.augmentations_mode,
        type=args.augmentations_type,
        input_size=args.input_size,
        crop_size=args.crop_size,
        validation=True,
    )

    print('transforms defined')

    dm = config_datasets(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        csv_paths=[args.train_path, args.validation_path, args.test_path],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        validation_transforms=validation_transforms,
        video_level=False,
        pin_memory=args.pin_memory,
        distributed=args.distributed,
        world_size=args.world_size,
        rank=rank,
    )
    dm.setup(stage="fit")
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    print('dataloaders defined')

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
        wandb.log({'learning_rate': scheduler.get_lr()[0]})
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
            torch.save(model.state_dict(), os.path.join(save_model_dir, "best-ckpt.pt"))

    cleanup()


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
    bce_running_loss = []
    pbar = tqdm(train_dataloader, desc=f"epoch {epoch}.", unit="iter")

    for batch, (x, id, y) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()

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
            # pass the batch through the classifier
            output = model.fc(model(x)).flatten()
            # mixed loss calculation
            bce_loss = criterion(output, y)
            loss = (
                bce_loss + BarlowTwinsLoss(z_real) + BarlowTwinsLoss(z_fake)
            )
            # get the logarithm of loss
            loss = loss.log()

        # mixed-precesion if given in arguments
        if fp16_scaler:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss.append(loss.detach().cpu().numpy())
        bce_running_loss.append(bce_loss.detach().cpu().numpy())
        # log mean loss for the last 10 batches:
        if (batch + 1) % 10 == 0:
            wandb.log({"train-steploss": np.mean(running_loss[-10:])})
            wandb.log({"train-bceloss": np.mean(bce_running_loss[-10:])})

    # scheduler
    scheduler.step()
    train_loss = np.mean(running_loss)
    wandb.log({"train-epoch-loss": train_loss})

    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, id, y in val_dataloader:
        x = x.cuda()
        y = y.cuda().unsqueeze(1)

        outputs = model.fc(model(x))
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


def find_free_port():
    # find a free port
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


if __name__ == "__main__":
    main()
