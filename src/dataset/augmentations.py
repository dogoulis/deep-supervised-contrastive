import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as TT


def get_gan_training_augmentations(aug_type, resize_size=256, crop_size=224):
    if "geometric" in aug_type:
        augmentations = [
            TT.RandomResizedCrop(
                resize_size, scale=(0.95, 1.0), ratio=(0.8, 1.2)
            )
        ]
    else:
        augmentations = [
            TT.Resize(resize_size)
        ]

    if "soft" in aug_type:
        pass
    elif "Wang" in aug_type:
        # add Wang augmentations pipeline transformed into albumentations:
        augmentations.extend(
            [
                TT.RandomApply([
                    TT.GaussianBlur(kernel_size=[3,5,7], sigma=(0.0, 3.0))
                ], p=0.5)
                # TODO
                # TT.RandomApply([
                #     TT.ImageCompression(quality_lower=30, quality_upper=100),
                # ], p=0.5),
            ]
        )
    elif "oneof" in aug_type:
        augmentations.append(
            TT.RandomChoice([
                TT.GaussianBlur(kernel_size=[3,5,7], sigma=(0.0, 3.0)),
                # TT.ImageCompression(quality_lower=30, quality_upper=100), # TODO
                # TT.ISONoise(intensity=0.1), # TODO
                TT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            ], p=[0.5, 0.5])
        )
    elif "strong" in aug_type:
        augmentations.append(
            TT.RandomChoice([
                TT.GaussianBlur(kernel_size=[3,5,7], sigma=(0.0, 3.0)),
                # TT.ImageCompression(quality_lower=30, quality_upper=100), # TODO
                # TT.ISONoise(intensity=0.1), # TODO
                TT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            ], p=[0.5, 0.5])
        )
        augmentations.append(
            TT.RandomChoice([
                TT.GaussianBlur(kernel_size=[3,5,7], sigma=(0.0, 3.0)),
                # TT.ImageCompression(quality_lower=30, quality_upper=100), # TODO
                # TT.ISONoise(intensity=0.1), # TODO
                TT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            ], p=[0.5, 0.5])
        )

    return A.Compose(
        augmentations
        + [
            TT.RandomCrop(crop_size),
            TT.HorizontalFlip(),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TT.ToTensor(),
        ]
    )


def get_gan_validation_augmentations(resize_size=256, crop_size=224):
    return TT.Compose(
        [
            TT.Resize(resize_size),
            TT.CenterCrop(crop_size),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TT.ToTensor(),
        ]
    )


def get_df_validation_augmentations(input_size=300, interpolation=cv2.INTER_LINEAR):
    return TT.Compose(
        [
            TT.Resize(input_size, interpolation=interpolation),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TT.ToTensor(),
        ]
    )


def get_df_training_augmentations(
    df_aug=None, input_size=300, interpolation=cv2.INTER_LINEAR
):
    if df_aug == "rand":
        return TT.Compose([
            TT.transforms.RandomChoice([
                TT.ColorJitter(brightness=0.2, contrast=0.2),
                TT.RandomRotation(30),
                # TT.RandomGamma(gamma_limit=(80, 120)), # TODO
                TT.RandomAdjustSharpness(0.2),
                TT.RandomAffine(degrees=(0.9, 1.1), translate=(0, 0.1))
            ]),
            TT.RandomChoice([
                TT.ColorJitter(brightness=0.2, contrast=0.2),
                TT.RandomRotation(30),
                # TT.RandomGamma(gamma_limit=(80, 120)), # TODO
                TT.RandomAdjustSharpness(0.2),
                TT.RandomAffine(degrees=(0.9, 1.1), translate=(0, 0.1))
            ]), 
            TT.Resize(input_size, interpolation=cv2.INTER_CUBIC),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TT.ToTensor(),
        ])
    elif df_aug == "validation":
        return get_df_validation_augmentations(input_size, interpolation)
    else:
        return ValueError("df_aug type not defined")