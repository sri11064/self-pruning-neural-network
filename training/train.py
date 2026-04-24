import torch
import torch.nn as nn
from training.loss import compute_sparsity_loss

def train_one_epoch(model, loader, optimizer, lambda_, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        ce_loss = nn.CrossEntropyLoss()(outputs, y)
        sparsity_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_ * sparsity_loss * 10

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)