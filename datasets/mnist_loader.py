from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, download=True, root='./data'):

    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    train_ds = datasets.MNIST(root = root, train = True, transform = transform, download = download)
    test_ds = datasets.MNIST(root = root, train = False, transform = transform, download = download)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2000, shuffle=False)

    return train_loader, test_loader

