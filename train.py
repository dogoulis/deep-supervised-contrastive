from random import shuffle
import numpy as np
from tqdm import tqdm
import os
import torch
from torch import nn
import wandb

import argparse

from torch.utils.data.dataloader import DataLoader
from dataset import gan_dataset, gan_aug

# Experiment Configuration:

config = {'projector': [2048] + [8192, 8192, 8192]}

# parser:
parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('--project_name', type=str, required=True,
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                    metavar='dataset_dir', help='Directory where the datasets are stored.')

parser.add_argument('-e', '--epochs', type=int, default=100,
                    metavar='epochs', help='Number of epochs.')

parser.add_argument('--fp16', type=str, default=None,
                    metavar='fp16', help='Indicator for using mixed precision.')

parser.add_argument('--save_dir', type=str,
                    metavar='save-dir', help='Save directory path.')

parser.add_argument('--name', type=str,
                    metavar='name', help='Experiment name that logs into wandb.')

parser.add_argument('--group', type=str,
                    metavar='group', help='Grouping argument for W&B init.')

parser.add_argument('--workers', type=str, default=12,
                    metavar='workers', help='Number of workers for the dataloader.')

parser.add_argument('--device', type=int, default=0,
                    metavar='device', help='Device used during training (default: 0).')

parser.add_argument('--train_dir', type=str,
                    metavar='train-dir', help='Training dataset path for csv.')

parser.add_argument('-b', '--batch_size', type=int, default=32,
                    metavar='batch_size', help='Input batch size for training (default: 32).')

parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    metavar='Learning Rate', help='Learning rate of the optimizer (default: 1e-3).')

parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                    metavar='Weight Decay', help='Weight decay of the optimizer (default: 1e-5).')

                    
args = parser.parse_args()

# define training logic:
def train_epoch(model, train_dataloader, args, optimizer, criterion, scheduler=None,
                fp16_scaler=None, epoch=0):

    # to train only the classification layer:

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
        
        # pass the real and fake batches through the backbone network and the through the projectors
        z_real = model.real_projector(model.backbone(real_class_batch))
        z_fake = model.fake_projector(model.backbone(fake_class_batch))
        
        # mixed loss calculation
        loss = criterion((model.bn(z_real), model.bn(z_fake)),
                            (y[real_idxs], y[fake_idxs]))
        
        # mixed-precesion if given in arguments
        if fp16_scaler is not None:
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
    
    train_loss = np.mean(running_loss)

    wandb.log({'train-epoch-loss': train_loss})
    
    return train_loss

# MAIN def
def main():

    # initialize weights and biases:

    wandb.init(project=args.project_name, name=args.name,
                config=vars(args), group = args.group, save_code=True)
    

    model = model.to(args.device)

    # define training transforms/augmentations:
    train_transforms = gan_aug.get_training_augmentations(args.aug)


    # set the path for training:
    train_dataset = gan_dataset.dataset2(args.dataset_dir, args.train_dir, train_transforms)

    # defining data loader:
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size,
                                    shuffle=True)

    # define optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # define the criterion:
    criterion = None

    fp16_scaler = None
    if args.fp16 is not None:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # directory:
    save_dir = args.save_dir
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define vale for min-loss:
    min_loss, train_results = float('inf'), {}
    print('Training starts...')

    for epoch in range(args.epochs):

        wandb.log({'epoch': epoch})
        train_results = train_epoch(model, train_dataloader=train_dataloader, args=args, optimizer=optimizer, criterion=criterion,
                    scheduler=None, fp16_scaler=fp16_scaler, epoch=epoch)

        if train_results['training-loss'] < min_loss:
            min_loss = train_results['training-loss'].copy()
            torch.save(model.state_dict(), os.path.join(save_dir, 'best-ckpt.pt'))

if __name__ == '__main__':
    main()