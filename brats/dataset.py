from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
from pathlib import Path

class BratsDataset(Dataset):
    def __init__(self, path, transform, test=False):
        self.path = path
        self.transform = transform
        self.files = []

        with open(self.path / 'test.txt' if test else self.path / 'train.txt') as f:
            for line in f.readlines():
                line = line.strip()
                self.files.append((line, 'L1'))
                self.files.append((line, 'L2'))
                self.files.append((line, 'L3'))

    def __getitem__(self, index):
        filename, layer = self.files[index]
        base_path = self.path / filename / layer
        toTensor = transforms.ToTensor()
        return {
            'filename': filename,
            'layer': layer,
            'input': self.transform(torch.stack([
                toTensor(Image.open(base_path / 't1.png')).squeeze(0),
                toTensor(Image.open(base_path / 't1ce.png')).squeeze(0),
                toTensor(Image.open(base_path / 't2.png')).squeeze(0),
                toTensor(Image.open(base_path / 'flair.png')).squeeze(0),
            ])),
            'segment': toTensor(Image.open(base_path / 'segment.png')).squeeze(0) / 255.0,
        }

    def __len__(self):
        return len(self.files)

def load_dataset(batch_size):
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    ])
    train_dataset = BratsDataset(Path('data/processed'), transform=transform, test=False)
    test_dataset = BratsDataset(Path('data/processed'), transform=transform, test=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
