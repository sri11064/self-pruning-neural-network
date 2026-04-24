import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from models.network import PrunableNet
from training.train import train_one_epoch
from utils.metrics import accuracy, calculate_sparsity
from utils.visualization import plot_gate_distribution
import config

device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)

# Proper normalization (important)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

results = []

for lambda_ in config.LAMBDA_VALUES:
    print(f"\n====== Training with lambda = {lambda_} ======")

    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, lambda_, device)
        acc = accuracy(model, test_loader, device)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc:.2f}%")

    # Final evaluation
    acc = accuracy(model, test_loader, device)
    sparsity = calculate_sparsity(model)

    print(f"Final Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"results/model_lambda_{lambda_}.pth")

    # Plot distribution
    plot_gate_distribution(model, f"results/gate_dist_lambda_{lambda_}.png")

    results.append((lambda_, acc, sparsity))

# Print summary
print("\n===== FINAL RESULTS =====")
for r in results:
    print(f"Lambda={r[0]} | Accuracy={r[1]:.2f}% | Sparsity={r[2]:.2f}%")