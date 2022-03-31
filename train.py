import argparse
import os
from random import shuffle
from wsgiref import validate

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.dataset import augmentations as aug
from src.dataset import datasets
from src.losses import BarlowTwinsLoss
from src.model import  Model

# CMD ARGUMENTS
parser = argparse.ArgumentParser(description='Training arguments')
# WANDB
parser.add_argument('-p', '--project_name', type=str, required=True,
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')
parser.add_argument('-rg', '--run-group', type=str, default=config['run_group'],
                    help='group of runs to put the current run into (e.g. ff)')
parser.add_argument('--name', type=str,
                    metavar='name', help='Experiment name that logs into wandb.')
# TRAINING
parser.add_argument('-e', '--epochs', type=int, default=10, required=True,
                    metavar='epochs', help='Number of epochs to train for')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    metavar='batch_size', help='Input batch size for training (default: 32).')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    metavar='Learning Rate', help='Learning rate of the optimizer (default: 1e-3).')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                    metavar='weight_decay', help='Weight decay of the optimizer (default: 1e-5).')
parser.add_argument('-sch', '--scheduler', type=str, default=None,
                    metavar='scheduler', help='Scheduler to use during training (default: None).')
# DATASET
parser.add_argument('-aug', '--augmentations', type=str, default=None,
                    metavar='augmentations', help='augmentations for the dataset')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    metavar='dataset', help='dataset on which to evaluate (default: None)')
parser.add_argument('-dp', '--dataset_path', type=str, default=None,
                    metavar='dataset', help='dataset on which to evaluate')
# MODEL DETAILS
parser.add_argument('-proj', '--projector', type=int, nargs='+', default=[2048] + [8192, 8192, 8192],
                    metavar='projector', help='projector architecture')
# PATHS
parser.add_argument('-save', '--save_model_path', type=str,
                    metavar='save_model_path', help='Save directory path for model.')
parser.add_argument('--save_back_path', type=str,
                    metavar='save_back_path', help='Save directory path for backbone net.')
parser.add_argument('--train_path', type=str,
                    metavar='train_path', help='Training dataset path for csv.')
parser.add_argument('--valid_path', type=str,
                    metavar='valid_path', help='Validation dataset path for csv.')
# OTHER
parser.add_argument('--device', type=int, default=0,
                    metavar='device', help='Device used during training (default: 0).')
parser.add_argument('-nw', '--num-workers', type=int, default=8, required=False,
                    metavar='num_workers', help='number of workers to use for dataloading (default: 8)')
parser.add_argument('-fp', '--fp16', default=True, action='store_true',
                    metavar='fp16', help='boolean for using mixed precision.')
args = parser.parse_args()
print(args)
print(vars(args))


def main():
    # initialize weights and biases
    wandb.init(project=args.project_name, name=args.name,
                config=vars(args), group = args.group, save_code=True)

    # model definition
    model = Model(config=args)
    model = model.to(args.device)

    # define training transforms/augmentations
    train_transforms = aug.get_gan_training_augmentations(args.aug)
    validation_transforms = aug.get_gan_validation_augmentations(args.aug)

    # set the path for training
    train_dataset = datasets.dataset2(args.dataset_dir, args.train_dir, train_transforms)
    val_dataset = datasets.dataset2(args.dataset_dir, args.valid_dir, validation_transforms)

    # defining data loader
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size,
                                    shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 

    # setting the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    # define the criterion:
    criterion = nn.BCEWithLogitsLoss()

    # set up fp16
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # checkpointing - directories
    print(args.save_model_path)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    print(args.save_backbone_path)
    if not os.path.exists(args.save_backbone_path):
        os.makedirs(args.save_backbone_path)
        
    # define value for min-loss
    min_loss = float('inf')

    print('Training starts...')
    for epoch in range(args.epochs):
        wandb.log({'epoch': epoch})
        train_epoch(model, train_dataloader=train_dataloader, args=args, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, fp16_scaler=fp16_scaler, epoch=epoch)
        val_results = validate(model, val_dataloader=val_dataloader, args=args, criterion=criterion)

        # TODO add more functionality here
        # e.g. better names for checkpoints and patience for early stopping
        if val_results['val_loss'] < min_loss:
            min_loss = val_results['val_loss'].copy()
            torch.save(model.state_dict(), os.path.join(save_model_dir, 'best-ckpt.pt'))


def train_epoch(model, train_dataloader, args, optimizer, criterion, scheduler=None, fp16_scaler=None, epoch=0):

    # to train only the classification layer
    model.train()
    epoch += 1
    running_loss = []
    pbar = tqdm(train_dataloader, desc=f'epoch {epoch}.', unit='iter')
    
    for batch, (x,y) in enumerate(pbar):
        
        x.to(args.device)
        y.to(args.device)

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
            output = model(model.fc(x))
            # mixed loss calculation
            loss = criterion(output, y) + BarlowTwinsLoss(z_real) + BarlowTwinsLoss(z_fake)
        
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
        if (batch+1) % 10 == 0:
            wandb.log({'train-steploss': np.mean(running_loss[-10:])})
    
    # scheduler
    scheduler.step()
        
    train_loss = np.mean(running_loss)

    wandb.log({'train-epoch-loss': train_loss})
    
    return train_loss

# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in val_dataloader:
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

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
    wandb.log({'validation-loss' : val_loss})
    acc = 100. * np.mean(y_true == y_pred)
    wandb.log({'validation-accuracy' : acc})
    return{'val_acc' : acc, 'val_loss' : val_loss}


if __name__ == '__main__':
    main()
