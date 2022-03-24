import numpy as np
from tqdm import tqdm

import wandb

import argparse



# parser:
parser = argparse.ArgumentParser(description='Training arguments')

parser.add_argument('--project_name', type=str, required=True,
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                    metavar='dataset_dir', help='Directory where the datasets are stored.')

parser.add_argument('-e', '--epochs', type=int, default=15,
                    metavar='epochs', help='Number of epochs')

parser.add_argument('--device', type=str,
                    metavar='device', help='Device used during training')


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


