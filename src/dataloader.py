from dataset import Dataset

import h5py
import numpy as np
from keras import utils
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class Dataloader :
    def __init__(self, file_path):
        self.file_path = file_path
        # Conversion numpy -> PIL -> Tensor
        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64, 64)),  # Redimensionnement
                        transforms.ToTensor(),
                                            ])
    # Load Data
    def load_data(self) :
        # To get the images and labels from file
        with h5py.File('Data/Galaxy10_DECals.h5', 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        return images, labels.astype(np.int64)
    

    def get_dataloaders(self, test_size=0.8, batch_size=32):
        """Crée les DataLoaders pour l'entraînement"""
        images, labels = self.load_data()
        dataset = Dataset(images, labels, self.transform)

        train_size = int((1 - test_size) * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader