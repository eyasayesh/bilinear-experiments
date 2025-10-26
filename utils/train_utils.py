from __future__ import annotations

import torch
from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.nn.modules.loss import _Loss

def train_one_epoch(model: Module,
                    train_loader: DataLoader,
                    optimizer: Optimizer,
                    criterion: _Loss,
                    device:str) -> float:
    """Train the model for one epoch
    Args:
        model: the neural network model
        train_loader: DataLoader for training data
        optimizer: optimizer for updating model parameters
        criterion: loss function
        device: device to run the training on (cpu or cuda)
    Returns:
        avg_loss: average loss over the epoch
    """

    model.train()
    total_loss = 0.0

    for data, target in tqdm(train_loader, desc="Training", leave=True):
        data, target = data.to(device), target.to(device)


        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model: Module,
             test_loader: DataLoader,
             criterion: _Loss,
             device:str) -> float:
    """Evaluate the model on the test dataset
    Args:
        model: the neural network model
        test_loader: DataLoader for test data
        criterion: loss function
        device: device to run the evaluation on (cpu or cuda)
    Returns:
        avg_loss: average loss over the test dataset
        accuracy: accuracy over the test dataset
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating", leave=True):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def save_checkpoint(model, path="checkpoints/model.pth"):
    torch.save(model.state_dict(), path)