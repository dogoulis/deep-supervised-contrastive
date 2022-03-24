import enum
import os
import argparse
import numpy as np
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import wandb

from torch.utils.data.dataloader import DataLoader
from dataset import pytorch_dataset, augmentations



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

        real_idxs = y == 0 
        fake_idxs = y == 1
        
        real_class_batch = x[real_idxs]
        fake_class_batch = x[fake_idxs]

        z_real = model.real_projector(model.backbone(real_class_batch))
        z_fake = model.fake_projector(model.backbone(fake_class_batch))

        loss = criterion((model.bn(z_real), model.bn(z_fake)),
                            (y[real_idxs], y[fake_idxs]))
        
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


