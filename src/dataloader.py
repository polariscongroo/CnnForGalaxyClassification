import torch
from dataset import Dataset

import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Subset

class Dataloader:
    def __init__(self, file_path):
        self.file_path = file_path

        # Transforms avec data augmentation pour l'entraînement
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])

        # Transforms simples pour la validation
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        ])

    def load_data(self):
        with h5py.File(self.file_path, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])
        return images, labels.astype(np.int64)

    def get_dataloaders(self, test_size=0.4, batch_size=16):
        images, labels = self.load_data()

        dataset = Dataset(images, labels, transform=None)  # temporaire sans transform

        train_size = int((1 - test_size) * len(dataset))
        val_size = len(dataset) - train_size

        train_indices, val_indices = torch.utils.data.random_split(
            range(len(dataset)), [train_size, val_size]
        )

        # Création des deux datasets avec leur propre transform
        train_dataset = Dataset(images[train_indices.indices], labels[train_indices.indices], self.train_transform)
        val_dataset = Dataset(images[val_indices.indices], labels[val_indices.indices], self.val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader
