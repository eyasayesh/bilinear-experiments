import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ModularAdditionDataset(Dataset):
    """
    A PyTorch Dataset for modular addition:
        (a + b) % modulus
    where a, b ∈ [0, modulus - 1]
    """

    def __init__(self, modulus: int = 113, device: str = "cpu", dtype=torch.long):
        self.modulus = modulus
        self.device = device
        self.dtype = dtype

        # Generate all pairs (a, b)
        a = torch.arange(modulus, device=device, dtype=dtype)
        b = torch.arange(modulus, device=device, dtype=dtype)
        A, B = torch.meshgrid(a, b, indexing="ij")  # shape: (modulus, modulus)
        C = (A + B) % modulus

        # Flatten integer tensors
        a_flat = A.flatten()
        b_flat = B.flatten()
        c_flat = C.flatten()

        # One-hot encode
        a_onehot = torch.nn.functional.one_hot(a_flat, num_classes=modulus)
        b_onehot = torch.nn.functional.one_hot(b_flat, num_classes=modulus)

        # Concatenate a and b → shape: (N, modulus*2)
        self.inputs = torch.cat([a_onehot, b_onehot], dim=1).float()
        self.targets = c_flat

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Returns:
            input: tensor([a, b])
            target: tensor(result)
        """
        return self.inputs[idx], self.targets[idx]
    
    @classmethod
    def get_dataloaders(cls, modulus: int = 113, 
                       batch_size: int = 64, 
                       train_test_split: float = 0.8,
                       device: str = "cpu", 
                       dtype=torch.long):
        """
        Returns train and test DataLoaders for the modular addition dataset.
        """
        dataset = cls(modulus=modulus, device=device, dtype=dtype)
        total_len = len(dataset)
        train_len = int(train_test_split * total_len)
        test_len = total_len - train_len

        train_set, test_set = random_split(dataset, [train_len, test_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

# Alias
def get_mod_add_dataloaders(modulus: int = 113, 
                            batch_size: int = 64, 
                            train_test_split: float = 0.8,
                            device: str = "cpu", 
                            dtype=torch.long):
    return ModularAdditionDataset.get_dataloaders(modulus, batch_size, train_test_split, device, dtype)

