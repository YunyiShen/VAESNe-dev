from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import os
from PIL import Image
import torch
from torchvision import transforms

# multimodal stuff
class multimodalDataset(Dataset):
    def __init__(self, *datasets):
        assert all(len(d) == len(datasets[0]) for d in datasets), "All datasets must be the same length"
        self.datasets = datasets
        self.num_modes = len(datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list of str): List of image file paths.
            transform (callable, optional): Transform to apply to each image.
        """
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
                                            ]) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([])