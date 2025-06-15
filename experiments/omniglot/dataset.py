# torch
import torch
import torch.optim
import torch.utils.data
# torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# built-in
import numpy as np

# Based on torchvision datasets
# def get_dataset(batch_size, augmentation, num_workers):
#     # Create transformations
#     # ----------------------
#     normalize = transforms.Normalize(mean=[0.5], std=[0.5])
#     resize_transform = transforms.Resize((28, 28)) #TODO: add a CNN layer to downsample instead of doing it manually
#     # Transformation for val and test
#     transf_test = transforms.Compose([resize_transform,
#                                       transforms.ToTensor(),
#                                       normalize,
#                                      ])
#     # Transformation for train
#     if augmentation:
#         transf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                         transforms.RandomCrop(32, 4),
#                                         transforms.ToTensor(),
#                                         normalize,
#                                         ])
#     else:
#         transf_train = transf_test
#     # ----------------------
#     # Download dataset and create dataloaders
#     train_loader = torch.utils.data.DataLoader(datasets.Omniglot(root='./data', transform=transf_train, download=True),
#                                                batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     # test_loader = torch.utils.data.DataLoader(datasets.Omniglot(root='./data', train=False, transform=transf_test),
#     #                                          batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     test_loader = None
#
#     dataloaders = {'train': train_loader,
#                    'validation': test_loader}
#     # Return dataloaders
#     return dataloaders, test_loader


def get_dataset(batch_size, augmentation, num_workers, test_split=0.2, root="./data"):
    # Create transformations
    # ----------------------
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    resize_transform = transforms.Resize((28, 28))  # TODO: add a CNN layer to downsample instead of doing it manually

    # Transformation for val and test
    transf_test = transforms.Compose([resize_transform,
                                      transforms.ToTensor(),
                                      normalize,
                                      ])

    # Transformation for train
    if augmentation:
        transf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop(32, 4),
                                           transforms.ToTensor(),
                                           normalize,
                                           ])
    else:
        transf_train = transf_test

    # ----------------------
    # Download full dataset
    full_dataset = datasets.Omniglot(root=root, transform=None, download=True)

    # Create deterministic train/test split
    dataset_size = len(full_dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    # Use torch.Generator for reproducible splits
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # Apply transforms to the split datasets
    # Create wrapper datasets that apply transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset_transformed = TransformDataset(train_dataset, transf_train)
    test_dataset_transformed = TransformDataset(test_dataset, transf_test)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataloaders = {'train': train_loader,
                   'validation': test_loader}

    # Return dataloaders
    return dataloaders, test_loader


