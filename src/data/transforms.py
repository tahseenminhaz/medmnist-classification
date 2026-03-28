"""Data augmentation and preprocessing transforms."""

from torchvision import transforms


def get_train_transforms(config: dict) -> transforms.Compose:

    aug = config.get("data", {}).get("augmentation", {})
    n_channels = config.get("data", {}).get("n_channels", 1)
    transform_list = []

    if aug.get("random_horizontal_flip"):
        transform_list.append(transforms.RandomHorizontalFlip())

    rotation = aug.get("random_rotation", 0)
    if rotation:
        transform_list.append(transforms.RandomRotation(rotation))

    cj = aug.get("color_jitter")
    if cj:
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(config: dict) -> transforms.Compose:
    n_channels = config.get("data", {}).get("n_channels", 1)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
    ])
