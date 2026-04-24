import torch

def compute_sparsity_loss(model):
    total = 0.0
    count = 0

    for gates in model.get_all_gates():
        total += torch.sum(gates)
        count += gates.numel()

    return total / count   