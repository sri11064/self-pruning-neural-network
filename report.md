# Self-Pruning Neural Network — Case Study Report

## 1. Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training.

Unlike traditional pruning (post-training), this approach integrates pruning directly into the learning process using **learnable gates**.

The model is trained on the **CIFAR-10 dataset** to evaluate the trade-off between **model accuracy and sparsity**.

---

## 2. Methodology

### 2.1 Prunable Linear Layer

A custom `PrunableLinear` layer was implemented instead of using `nn.Linear`.

Each weight is associated with a learnable **gate parameter**:

$$
g = \sigma(\text{gate_scores})
$$

$$
W_{pruned} = W \cdot g
$$

Where:

* ( \sigma ) is the sigmoid function
* ( g \in (0,1) ) controls whether a weight is active or pruned

This allows the network to **learn which connections are important**.

---

### 2.2 Sparsity Regularization

To encourage pruning, an **L1 penalty on gate values** is added:

$$
\text{Total Loss} = \text{CrossEntropy} + \lambda \cdot \sum g
$$

#### Why L1 works:

* L1 promotes **exact zeros** (sparsity)
* It creates a **sharp constraint geometry**
* Unlike L2, it does not just shrink values — it encourages elimination

---

### 2.3 Training Setup

* Dataset: CIFAR-10
* Optimizer: Adam
* Batch Size: 128
* Epochs: 10
* Loss: CrossEntropy + λ × Sparsity Loss

Three values of λ were tested:

* λ = 0.01 (moderate pruning pressure)
* λ = 0.1 (higher pruning pressure)
* λ = 1.0 (strong pruning pressure)

---

## 3. Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 0.01       | 52.35             | 0.00         |
| 0.1        | 52.06             | 0.00         |
| 1.0        | 52.65             | 0.00         |

### Observations:

* The model achieves stable accuracy (~52–55%) across different λ values
* However, sparsity remains **0% for all experiments**
* This indicates that the sparsity regularization was insufficient to influence the optimization process

This behavior suggests that the classification objective dominates the loss, preventing gate values from moving toward zero.

---

## 4. Gate Distribution Analysis

A histogram of gate values was plotted after training.

### Key Observation:

* Gate values remain concentrated near **1**
* Very few values approach **0**

This indicates that the model has **not yet learned to separate important and redundant connections**, and pruning has not effectively occurred.

---

## 5. Sparsity Definition

A weight is considered **pruned** if:

$$
g < 10^{-2}
$$

Sparsity (%) is computed as:

$$
\frac{\text{Number of pruned weights}}{\text{Total weights}} \times 100
$$

---

## 6. Key Insights

1. Simply adding L1 regularization is **not sufficient** to induce sparsity
2. The relative scale of sparsity loss vs classification loss is critical
3. Proper tuning of λ or scaling of sparsity term is required
4. Training duration also impacts pruning effectiveness

---

## 7. Real-World Impact

This approach is highly relevant for:

* Edge devices (low memory environments)
* Mobile AI applications
* Low-latency inference systems
* Model compression pipelines

However, achieving practical benefits requires **effective sparsity learning**, which depends on proper loss balancing.

---

## 8. Limitations

* Fully connected network limits performance on CIFAR-10
* No effective pruning observed in current configuration
* Sparsity loss scaling is insufficient
* Hyperparameter tuning (λ, scaling factor) is required

---

## 9. Future Work

* Increase sparsity pressure (scale L1 term)
* Train for more epochs
* Extend to **CNN-based architectures**
* Apply structured pruning (channel/filter-level)
* Integrate with **FastAPI for deployment**
* Combine with **quantization techniques**

---

## 10. Conclusion

This project successfully implements the **mechanism of self-pruning using learnable gates and L1 regularization**.

While current results do not exhibit effective sparsity, the framework highlights the importance of **loss balancing and hyperparameter tuning** in achieving meaningful model compression.

The approach provides a strong foundation for building efficient neural networks when properly optimized.

---
