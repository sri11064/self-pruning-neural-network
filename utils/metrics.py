import torch

def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for gates in model.get_all_gates():
        total += gates.numel()
        pruned += (gates < threshold).sum().item()

    return (pruned / total) * 100


def accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

def apply_hard_pruning(model, threshold=1e-2):
    """
    Converts soft gates into hard pruning by zeroing out weights
    whose gate values are below the threshold.
    """
    with torch.no_grad():
        for layer in [model.fc1, model.fc2, model.fc3]:
            gates = torch.sigmoid(layer.gate_scores)
            mask = (gates >= threshold).float()

            # Apply mask to weights
            layer.weight.data *= mask