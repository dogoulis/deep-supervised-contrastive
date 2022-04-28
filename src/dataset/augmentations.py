import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def get_gan_training_augmentations(aug_type, resize_size=256, crop_size=224):
    if "geometric" in aug_type:
        augmentations = [
            A.augmentations.crops.transforms.RandomResizedCrop(
                resize_size, resize_size, (0.95, 1.0), (0.8, 1.2)
            )
        ]
    else:
        augmentations = [
            A.augmentations.geometric.resize.Resize(resize_size, resize_size)
        ]

    if "soft" in aug_type:
        pass
    elif "Wang" in aug_type:
        # add Wang augmentations pipeline transformed into albumentations:
        augmentations.extend(
            [
                A.augmentations.transforms.GaussianBlur(sigma_limit=(0.0, 3.0), p=0.5),
                A.augmentations.transforms.ImageCompression(
                    quality_lower=30, quality_upper=100, p=0.5
                ),
            ]
        )
    elif "oneof" in aug_type:
        augmentations.append(
            A.OneOf(
                [
                    A.augmentations.transforms.GaussianBlur(
                        sigma_limit=(0.0, 3.0), p=0.5
                    ),
                    A.augmentations.transforms.ImageCompression(
                        quality_lower=30, quality_upper=100, p=0.5
                    ),
                    A.augmentations.transforms.ISONoise(p=0.5),
                    A.augmentations.transforms.ColorJitter(0.4, 0.4, 0.0, 0.0, p=0.5),
                ]
            )
        )
    elif "strong" in aug_type:
        augmentations.append(
            A.SomeOf(
                [
                    A.augmentations.transforms.GaussianBlur(
                        sigma_limit=(0.0, 3.0), p=0.5
                    ),
                    A.augmentations.transforms.ImageCompression(
                        quality_lower=30, quality_upper=100, p=0.5
                    ),
                    A.augmentations.transforms.ISONoise(p=0.5),
                    A.augmentations.transforms.ColorJitter(0.4, 0.4, 0.0, 0.0, p=0.5),
                ],
                2,
            )
        )

    return A.Compose(
        augmentations
        + [
            A.augmentations.crops.transforms.RandomCrop(crop_size, crop_size),
            A.augmentations.transforms.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_gan_validation_augmentations(resize_size=256, crop_size=224):
    return A.Compose(
        [
            A.augmentations.geometric.resize.Resize(resize_size, resize_size),
            A.augmentations.crops.transforms.CenterCrop(crop_size, crop_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_df_validation_augmentations(input_size=300, interpolation=cv2.INTER_LINEAR):
    return A.Compose(
        [
            A.Resize(input_size, input_size, interpolation=interpolation),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_df_training_augmentations(
    df_aug=None, input_size=300, interpolation=cv2.INTER_LINEAR
):
    if df_aug == "validation":
        return A.Compose(
            [
                A.Resize(input_size, input_size, interpolation=interpolation),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    elif df_aug == "train":
        return A.Compose(
            [
                A.Resize(input_size, input_size, interpolation=interpolation),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return ValueError("df_aug type not defined")
