import copy

import timm
from torch import nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["model"] == 'resnet50':
            self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
            self.head = nn.Linear(2048, config['num_head_neurons'])
        elif config["model"] == 'vit_small':
            self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
            self.head = nn.Linear(384, config['num_head_neurons'])
        elif config["model"] == 'vit_small_21k':
            self.backbone = timm.create_model("vit_small_patch16_224_in21k", pretrained=True, num_classes=0)
            self.head = nn.Linear(384, config['num_head_neurons'])
        else:
            raise ValueError(f"Model {config['model']} is not supported")

        # projector:
        projector_layers = []
        for i in range(len(config["projector"]) - 2):
            projector_layers.append(
                nn.Linear(
                    config["projector"][i], config["projector"][i + 1], bias=False
                )
            )
            projector_layers.append(nn.BatchNorm1d(config["projector"][i + 1]))
            projector_layers.append(nn.ReLU(inplace=True))
        projector_layers.append(
            nn.Linear(config["projector"][-2], config["projector"][-1], bias=False)
        )

        # one projection head for each class:
        self.real_projector = nn.Sequential(*projector_layers)
        self.fake_projector = copy.deepcopy(self.real_projector)
        self.synthetic_projector = copy.deepcopy(self.real_projector)

        # normalization layer for the representations of z1 and z2:
        self.bn = nn.BatchNorm1d(config["projector"][-1], affine=False)

    def forward(self, x):
        return self.backbone(x)
