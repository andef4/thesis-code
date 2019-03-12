import csv
import os
import math
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
from collections import defaultdict


CSV_FILE = os.path.join('..', 'data', 'Data_Entry_2017.csv')
BATCH_SIZE = 10


class XRayDataset(Dataset):
    def __init__(self, transform, validation=False):
        self.transform = transform
        self.files = []
        if not os.path.exists(CSV_FILE):
            raise Exception('missing csv data file {}, please download data as described in README.md'.format(CSV_FILE))

        self.classes = set()
        self.class_counts = defaultdict(lambda: 0)

        with open('../data/test_list.txt' if validation else '../data/train_val_list.txt') as f:
            filenames = set([s.strip() for s in f.readlines()])

        with open(CSV_FILE) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader) # skip header
            for row in reader:
                filename, labels, *_ = row
                if filename not in filenames:
                    continue

                labels = labels.split('|')
                # only use images with a single class
                if len(labels) != 1 or labels[0] == 'No Finding':
                    continue
                self.files.append((filename, labels[0]))
                self.classes.update(labels)
                self.class_counts[labels[0]] += 1

        # convert set to list to have a guaranteed iteration order
        # this should also be the case with a set, but it is not explictly defined
        self.classes = sorted(list(self.classes))
        class_weights = []
        for class_ in self.classes:
            class_weights.append(1 / self.class_counts[class_])
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def __getitem__(self, index):
        filename, label = self.files[index]
        image = Image.open(os.path.join('..', 'data', 'processed_images', filename))
        image = self.transform(image)
        return image, torch.tensor(self.classes.index(label), dtype=torch.long)

    def __len__(self):
        return len(self.files)

def load_model(device):
    dataset = XRayDataset(transform=transforms.ToTensor(), validation=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    num_classes = len(dataset.classes)

    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
    state_dict = torch.load('../models/densenet_single_full_nonofindings_weighted_40_0.36186.pth')
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    return dataset, loader, model
