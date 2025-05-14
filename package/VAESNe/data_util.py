from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

class multimodalDataset(Dataset):
    def __init__(self, d1, d2):
        assert len(d1) == len(d2), "Datasets must be the same length"
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)

    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx]