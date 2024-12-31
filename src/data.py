# Load in relevant libraries, and alias where appropriate
import torch
import torchvision
import torchvision.transforms as transforms



# Use transforms.compose method to reformat images for modeling,
# and save to variable preprocessing_stepss for later use
DEFAULT_PREPROCESSING_STEPS = transforms.Compose(
    [
        transforms.Resize((32,32)),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], # FYI: The mean of the training dataset for your convenience
            std=[0.2023, 0.1994, 0.2010]  # FYI the standard deviation of the training dataset
        )
    ]
)


def load_cifar10_data(
    prep_steps: "transforms.Compose" = None,
    batch_size: int = 64,
    shuffle: bool = True,
    root: str = '../data',
    download: bool = True,
    verbose: bool = True,
    validation_split: float = 0.2
):
    """ Load in the CIFAR10 dataset from torchvision.datasets.CIFAR10, preprocess the data,
    and create dataloaders for training, validation, and testing datasets.

    Args:
        prep_steps (transforms.Compose): preprocessing steps in a pytorch transforms.Compose object
        batch_size (int): number of images to process at once
        shuffle (bool): shuffle the dataset (i.e. randomize the order of the images)
        root (str): where the data is stored if downloaded
        download (bool): download the data if it is not already present
        verbose (bool): print out information about the dataset
        validation_split (float): proportion of the training data to use for validation

    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training dataset
        val_loader (torch.utils.data.DataLoader): dataloader for validation dataset
        test_loader (torch.utils.data.DataLoader): dataloader for testing dataset
        train_dataset (torchvision.datasets.CIFAR10): training dataset
        val_dataset (Subset): validation dataset (subset of training data)
        test_dataset (torchvision.datasets.CIFAR10): testing dataset
    """
    # Download, preprocess, and instantiate training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=prep_steps,
        download=download
    )

    # Split training data into training and validation datasets
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Download, preprocess, and instantiate testing dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=prep_steps,
        download=download
    )

    # Instantiate loader objects to facilitate processing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle validation data
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle test data
    )

    if verbose:
        print(f"\nTraining dataloader contains {len(train_loader)} batches of {batch_size} images - In total {len(train_dataset)} images")
        print(f"Validation dataloader contains {len(val_loader)} batches of {batch_size} images - In total {len(val_dataset)} images")
        print(f"Testing dataloader contains {len(test_loader)} batches of {batch_size} images - In total {len(test_dataset)} images")
        print(f"Classes in the dataset: {test_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
