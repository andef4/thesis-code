from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image


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
        return {
            'filename': filename,
            'layer': layer,
            'input': torch.stack([
                self.transform(Image.open(base_path / 't1.png')).squeeze(0),
                self.transform(Image.open(base_path / 't1ce.png')).squeeze(0),
                self.transform(Image.open(base_path / 't2.png')).squeeze(0),
                self.transform(Image.open(base_path / 'flair.png')).squeeze(0),
            ]),
            'segment': transforms.ToTensor()(Image.open(base_path / 'segment.png')).squeeze(0) / 255.0,
        }

    def __len__(self):
        return len(self.files)
