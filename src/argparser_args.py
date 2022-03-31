import argparse

def get_argparser():
    # CMD ARGUMENTS
    parser = argparse.ArgumentParser(description='Training arguments')
    # WANDB
    parser.add_argument('-p', '--project_name', type=str, default=None, required=True,
                        metavar='project_name', help='Project name, utilized for logging purposes in W&B.')
    parser.add_argument('-rg', '--run-group', type=str, default=None, required=False,
                        help='group of runs to put the current run into (e.g. ff)')
    parser.add_argument('--name', type=str, default=None, required=False,
                        metavar='name', help='Experiment name that logs into wandb.')
    # TRAINING
    parser.add_argument('-e', '--epochs', type=int, default=10, required=False,
                        metavar='epochs', help='Max number of epochs to train for')
    parser.add_argument('-b', '--batch_size', type=int, default=32, required=False,
                        metavar='batch_size', help='Input batch size for training (default: 32).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, required=False,
                        metavar='learning_rate', help='Learning rate of the optimizer (default: 1e-3).')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, required=False,
                        metavar='weight_decay', help='Weight decay of the optimizer (default: 1e-5).')
    parser.add_argument('-sch', '--scheduler', type=str, default='step_rl', required=False,
                        metavar='scheduler', help='Scheduler to use during training (default: None).')
    # DATASET
    parser.add_argument('-aug', '--augmentations', type=str, default=None, required=True,
                        metavar='augmentations', help='augmentations for the dataset')
    parser.add_argument('-d', '--dataset', type=str, default=None, required=True,
                        metavar='dataset', help='dataset on which to evaluate')
    parser.add_argument('-dp', '--dataset_path', type=str, default=None, required=True,
                        metavar='dataset', help='dataset on which to evaluate')
    # DATA PATHS
    parser.add_argument('-tp', '--train_path', type=str, default=None, required=False,
                        metavar='train_path', help='Training dataset path for csv.')
    parser.add_argument('-vp', '--validation_path', type=str, default=None, required=False,
                        metavar='validdation_path', help='Validation dataset path for csv.')
    # MODEL DETAILS
    parser.add_argument('-proj', '--projector', type=int, nargs='+', default=[2048] + [8192, 8192, 8192], required=False,
                        metavar='projector', help='projector architecture')
    # CHECKPOINT PATHS
    parser.add_argument('-savem', '--save_model_path', type=str, default='./checkpoints/model', required=False,
                        metavar='save_model_path', help='Save directory path for model.')
    parser.add_argument('--saveb', '--save_backbone_path', type=str, default='./checkpoints/backbone', required=False,
                        metavar='save_backbone_path', help='Save directory path for backbone net.')
    # OTHER
    parser.add_argument('--device', type=int, default=None, required=True,
                        metavar='device', help='Device used during training')
    parser.add_argument('-nw', '--num-workers', type=int, default=8, required=False,
                        metavar='num_workers', help='number of workers to use for dataloading (default: 8)')
    parser.add_argument('-fp', '--fp16', default=True, action='store_true', required=False,
                        help='boolean for using mixed precision.')
    
    return parser