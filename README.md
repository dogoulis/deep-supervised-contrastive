# deep-supervised-contrastive
Research Code for deepfake & gan classification.

## train.py usage:
```
usage: train.py [-h] -p project_name [-rg RUN_GROUP] [--name name] [-e epochs] [-b batch_size] [-opt optimizer] [-lr learning_rate] [-wd weight_decay] [-sch scheduler] [-step scheduler_step_size]
                [-gamma scheduler_gamma] -aug augmentations -d dataset -dp dataset [-tp train_path] [-vp validdation_path] [-proj projector [projector ...]] [-savem save_model_path]
                [--saveb save_backbone_path] --device device [-nw num_workers] [-fp]

Training arguments

optional arguments:
  -h, --help            show this help message and exit
  -p project_name, --project_name project_name
                        Project name, utilized for logging purposes in W&B.
  -rg RUN_GROUP, --run-group RUN_GROUP
                        group of runs to put the current run into (e.g. ff)
  --name name           Experiment name that logs into wandb.
  -e epochs, --epochs epochs
                        Max number of epochs to train for
  -b batch_size, --batch_size batch_size
                        Input batch size for training (default: 32).
  -opt optimizer, --optimizer optimizer
                        optimizer to use during training (default: adam).
  -lr learning_rate, --learning_rate learning_rate
                        Learning rate of the optimizer (default: 1e-3).
  -wd weight_decay, --weight_decay weight_decay
                        Weight decay of the optimizer (default: 1e-5).
  -sch scheduler, --scheduler scheduler
                        Scheduler to use during training (default: steprl).
  -step scheduler_step_size, --scheduler_step_size scheduler_step_size
                        scheduler step size (default: 5)
  -gamma scheduler_gamma, --scheduler_gamma scheduler_gamma
                        scheduler gamma (default: 0.1)
  -aug augmentations, --augmentations augmentations
                        augmentations for the dataset
  -d dataset, --dataset dataset
                        dataset on which to evaluate
  -dp dataset, --dataset_path dataset
                        dataset on which to evaluate
  -tp train_path, --train_path train_path
                        Training dataset path for csv.
  -vp validdation_path, --validation_path validdation_path
                        Validation dataset path for csv.
  -proj projector [projector ...], --projector projector [projector ...]
                        projector architecture
  -savem save_model_path, --save_model_path save_model_path
                        Save directory path for model.
  --saveb save_backbone_path, --save_backbone_path save_backbone_path
                        Save directory path for backbone net.
  --device device       Device used during training
  -nw num_workers, --num-workers num_workers
                        number of workers to use for dataloading (default: 8)
  -fp, --fp16           boolean for using mixed precision.
```
