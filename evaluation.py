import argparse
from distutils.command.config import config
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import Model
import torch
import torch.nn as nn
import wandb

from dataset import pytorch_dataset, augmentations
from torch.utils.data.dataloader import DataLoader
from torchmetrics import functional as tmf

# experiment config
config = {
            'projector' : [2048] + [8196, 8196, 8196]
}


# parser:
parser = argparse.ArgumentParser(description='Testing arguments')

parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                    metavar='dataset_dir', help='Directory where the datasets are stored.')

parser.add_argument('--device', type=int, default=0,
                    metavar='device', help='device used during training (default: 0)')

parser.add_argument('--test_dir', type=str,
                    metavar='testing-directory', help='Directory of the testing csv')

parser.add_argument('--id', type=int,
                    metavar='id', help='id of the test')

parser.add_argument('--weights_dir', type=str,
                    metavar='weights_dir', help='Directory of weights')

parser.add_argument('--name', type=str,
                    metavar='name', help='Experiment name that logs into wandb')

parser.add_argument('--project_name', type=str,
                    metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

parser.add_argument('--group', type=str,
                    metavar='group', help='Grouping argument for W&B init.')

parser.add_argument('--workers', default=8,
                    metavar='workers', help='Number of workers for the dataloader')

parser.add_argument('-b', '--batch_size', type=int, default=32,
                    metavar='batch_size', help='input batch size for training (default: 32)')


args = parser.parse_args()


@torch.no_grad()
def testing(model, dataloader, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in tqdm(dataloader):
        x = x.to(args.device)
        y = y.to(args.device).unsqueeze(1)

        outputs = model.fc(model(x))
        loss = criterion(outputs, y)

        running_loss.append(loss.cpu().numpy())
        outputs = torch.sigmoid(outputs)
        y_true.append(y.squeeze(1).cpu().int())
        y_pred.append(outputs.squeeze(1).cpu())
    wandb.log({'Loss': np.mean(running_loss)})

    return np.mean(running_loss), torch.cat(y_true, 0), torch.cat(y_pred, 0)


def log_metrics(y_true, y_pred):

    test_acc = tmf.accuracy(y_pred, y_true)
    test_f1 = tmf.f1(y_pred, y_true)
    test_prec = tmf.precision(y_pred, y_true)
    test_rec = tmf.recall(y_pred, y_true)
    test_auc = tmf.auroc(y_pred, y_true)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec,
        'ROC-AUC score': test_auc})


# main def:
def main():

    # initialize w&b
    print(args.name)
    wandb.init(project=args.project_name, name=args.name,
               config=vars(args), group=args.group)

    # initialize model:
    model = Model(config=config)

    # load weights:
    model.load_state_dict(torch.load(args.weights_dir, map_location='cpu'))

    model = model.eval().to(args.device)

    # defining transforms:
    transforms = augmentations.get_validation_augmentations()

    # define test dataset:
    test_dataset = pytorch_dataset.dataset2(
        args.dataset_dir, args.test_dir, transforms)

    # define data loaders:
    test_dataloader = DataLoader(test_dataset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    # set the criterion:
    criterion = nn.BCEWithLogitsLoss()

    # testing
    test_loss, y_true, y_pred = testing(
        model=model, dataloader=test_dataloader, criterion=criterion)

    # calculating and logging results:
    log_metrics(y_true=y_true, y_pred=y_pred)

    print(f'Finished Testing with test loss = {test_loss}')


if __name__ == '__main__':
    main()