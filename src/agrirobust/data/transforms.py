"""Image transformations and augmentation pipelines for AgriRobust."""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard augmentations for training: resized crop, horizontal flip, slight rotation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transformations for validation and test evaluation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
