"""
Training loop from https://github.com/andef4/deeplearning (MIT licensed), original source PyTorch tutorial:
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code, under BSD license

"""
import os
import torch
from pathlib import Path
from datetime import datetime


def train_model(name, model, dataloaders, criterion, optimizer, device, num_epochs=25):
    best_acc = 0.0

    if not os.path.exists('results'):
        os.mkdir('results')
    results_file = Path(f'results/{name}.txt')
    if results_file.exists():
        raise Exception('Result file already exists, please change name')
    f = open(results_file, 'w', buffering=1)

    for epoch in range(num_epochs):
        epoch_start = datetime.now()

        train_loss = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['input']
                inputs = inputs.to(device)
                labels = data['segment']
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss = epoch_loss
            else:
                time = datetime.now() - epoch_start
                stats = f'Epoch: {epoch}, TL: {train_loss:.5f}, VL: {epoch_loss:.5f}, Time: {time}'
                print(stats)
                f.write(f'{stats}\n')

            # save model to disk
            if phase == 'val':
                torch.save(model.state_dict(), f'models/{name}_{epoch}.pth')
