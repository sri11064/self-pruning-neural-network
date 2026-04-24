# Self-Pruning Neural Network 

## Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training.

Unlike traditional pruning (performed after training), this model integrates **learnable gating mechanisms** that dynamically suppress unimportant connections during optimization.

The implementation is designed with a **modular and production-oriented structure** to reflect real-world ML engineering practices.

The objective is to demonstrate:

* Deep learning fundamentals
* Custom PyTorch module design
* Trade-off analysis between accuracy and sparsity
* Clean, modular engineering practices

---

## Key Features

### 1. Custom Prunable Linear Layer

* Fully implemented without using `torch.nn.Linear`
* Each weight is associated with a **learnable gate parameter**
* Gates are constrained to (0,1) using a sigmoid function

---

### 2. Self-Pruning Mechanism

Weights are modulated using learnable gates:

$$
W_{pruned} = W \cdot \sigma(gate_scores)
$$

* Allows the network to learn which connections are important
* Enables dynamic suppression of less useful weights

---

### 3. Sparsity-Aware Loss Function

Total Loss:

$$
\text{Loss} = \text{CrossEntropy} + \lambda \cdot \text{Sparsity Loss}
$$

* Sparsity Loss uses **L1 regularization on gate values**
* Encourages gates to move toward zero (pruning)
* Introduces a trade-off between accuracy and sparsity

---

### 4. Trade-off Analysis (λ Experiments)

Experiments conducted with:

* λ = 0.001 → Low pruning pressure
* λ = 0.01 → Moderate pruning pressure
* λ = 0.1 → Higher pruning pressure

---

## Hard Pruning (Deployment Simulation)

In addition to soft gating, the model performs **hard pruning**:

* Weights with gate values < 1e-2 are permanently zeroed out
* This simulates **real-world deployment conditions**

Two metrics are reported:

* **Soft Accuracy** (before pruning)
* **Pruned Accuracy** (after pruning)

This ensures the model is not only trained with sparsity awareness but also evaluated in a **deployment-like setting**.

---

## Project Structure

```
self-pruning-network/
│
├── models/          # Custom layers & architecture
├── training/        # Loss & training loop
├── utils/           # Metrics & visualization
├── results/         # Saved models & plots
│
├── main.py          # Entry point
├── config.py        # Hyperparameters
├── report.md        # Detailed analysis
├── README.md        # Project documentation
```

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

---

## Results

| Lambda | Soft Accuracy (%) | Pruned Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------------- | ------------ |
| 0.01   | 54.86             | 52.35               | 0.00         |
| 0.1    | 55.17             | 52.06               | 0.00         |
| 1.0    | Not Available     | 52.65               | 0.00         |

### Observations

* The model achieves stable accuracy (~55%) across different λ values
* However, sparsity remains at **0%**, indicating ineffective pruning
* This suggests that the sparsity regularization is too weak relative to the classification objective

This experiment demonstrates that **simply adding L1 regularization is not sufficient** — proper scaling of the sparsity term or longer training is required to induce meaningful pruning.

---

## Visualization

The gate value distribution does not yet exhibit strong bimodal behavior.

* Most gate values remain close to 1
* Very few (or none) approach zero

This indicates that the model has **not yet learned to separate important and redundant connections**, reinforcing the need for stronger sparsity pressure.

---

## Engineering Highlights

* Modular and scalable project structure
* Custom neural network layer implementation
* Clear separation of concerns (models, training, utils)
* Config-driven experimentation
* Model checkpointing and logging

---

## Performance Considerations

* Supports **GPU acceleration (CUDA)** when available
* Efficient batching using PyTorch DataLoader
* Normalized sparsity loss for stable optimization

---

## Real-World Impact

This approach enables:

* Reduced model size
* Faster inference
* Lower memory footprint

Ideal for:

* Edge devices
* Mobile applications
* Low-latency AI systems

Instead of pruning after training, models can be **trained to be efficient from the start**.

---

## Future Improvements

* Increase sparsity pressure (scaled L1 penalty)
* Train for longer epochs
* Extend to CNN-based architectures
* Apply structured pruning (channels/filters)
* Integrate FastAPI for deployment
* Combine with quantization techniques

---

## Conclusion

This project demonstrates the implementation of a **self-pruning neural network using learnable gates and L1 regularization**.

While current results show limited sparsity, the framework correctly captures the mechanism and highlights the importance of **loss balancing and hyperparameter tuning** in achieving effective model compression.

---
