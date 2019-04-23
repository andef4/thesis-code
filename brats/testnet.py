from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from pathlib import Path
from PIL import Image


class TestnetDataset(Dataset):
    def __init__(self, path, transform, test=False):
        self.path = path
        self.transform = transform
        self.files = []

        files = sorted(os.listdir(path))
        if test:
            self.files = files[int(len(files) * 0.8):]
        else:
            self.files = files[:int(len(files) * 0.8)]

    def __getitem__(self, index):
        filename = self.files[index]
        return self.get_sample(filename)

    def get_sample(self, filename):
        base_path = self.path / filename
        toTensor = transforms.ToTensor()
        return {
            'filename': filename,
            'input': toTensor(Image.open(base_path / 'image.png')),
            'segment': toTensor(Image.open(base_path / 'segment.png')).squeeze(0) / 255.0,
        }

    def __len__(self):
        return len(self.files)


def load_dataset(batch_size):
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    ])
    train_dataset = TestnetDataset(Path('testnet'), transform=transform, test=False)
    test_dataset = TestnetDataset(Path('testnet'), transform=transform, test=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
