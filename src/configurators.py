from torch import nn, optim
import torch_optimizer


def config_optimizers(params, args):
    optimizer = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(params, lr=config["lr"])
    # elif args.optimizer == 'ranger':
    #     optimizer = torch_optimizer.Ranger(params,
    #                             lr=args.learning_rate,
    #                             weight_decay=args.weight_decay)
    return optimizer


def config_schedulers(optimizer, args):
    scheduler = None
    if args.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
        )
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
            verbose=True,
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, verbose=True
        )
    return scheduler
