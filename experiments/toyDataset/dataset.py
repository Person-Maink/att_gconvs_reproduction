# torch
import torch
import torch.optim
import torch.utils.data
# torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# built-in
import numpy as np
import os, json
from PIL import Image
from torch.utils.data import Dataset


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
def visualize_batch(batch, *, n_cols: int = 8) -> None:
    import matplotlib.pyplot as plt
    imgs = batch[0]
    imgs = imgs.detach().cpu()
    # emb  = emb.detach().cpu().float()
    # emb = emb.permute(0, 2, 1).reshape(emb.shape[0], -1)  # (B, M×16)

    B = imgs.size(0)
    n_rows = (B + n_cols - 1) // n_cols
    fig_imgs, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')

    for i in range(B):
        img = imgs[i]
        if img.min() < 0:                         # de-normalize if necessary
            img = (img * 0.5) + 0.5
        axes[i].imshow(img.squeeze(0), cmap='gray')


    plt.tight_layout()
    # plt.imshow(emb2d, aspect='auto')
    plt.show()


class JSONImageDataset(Dataset):
    """Images listed in a JSON file produced by the previous script:
       {"img_0000_label_0.png": 0, "img_0001_label_0.png": 0, ...}"""
    def __init__(self, json_path, root="", transform=None):
        with open(json_path, "r") as f:
            mapping = json.load(f)                       # filename → label

        self.items = [
            (os.path.join(root, fname), int(label))
            for fname, label in mapping.items()
        ]
        self.transform = transform
        self.num_classes = len({lbl for _, lbl in self.items})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataset(batch_size, augmentation, num_workers, test_split=0.2, root="./data"):
    from dataset import TransformDataset
    # Create transformations
    # ----------------------
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    resize_transform = transforms.Resize((64, 64))  # TODO: add a CNN layer to downsample instead of doing it manually

    # Transformation for val and test
    transf_test = transforms.Compose([resize_transform,
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])

    # Transformation for train
    if augmentation:
        transf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop(32, 4),
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.ToTensor(),
                                           normalize,
                                           ])
    else:
        transf_train = transf_test

    # ----------------------
    # Download full dataset
    # full_dataset = datasets.Omniglot(root=root, transform=None, download=True)
    full_dataset = JSONImageDataset(
        json_path=os.path.join(root, "data/file_to_label.json"),
        root="data",  # same folder that holds “data/000000.png”
        transform=None
    )

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


    train_dataset_transformed = TransformDataset(train_dataset, transf_train)
    test_dataset_transformed = TransformDataset(test_dataset, transf_test)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True
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


