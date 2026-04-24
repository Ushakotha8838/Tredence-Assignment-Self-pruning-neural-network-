# Self-Pruning Neural Network Report

## Why L1 Regularization Encourages Sparsity
L1 regularization penalizes the absolute values of gate parameters. This pushes many gate values toward zero, effectively turning off less important connections and making the network sparse.

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.01   | ~40%     | ~0%      |
| 1.0    | ~38%     | ~5%      |
| 20.0   | ~39%     | ~68%     |

---

## Analysis
As lambda increases, sparsity increases because more connections are pushed toward zero. However, very high sparsity can slightly reduce accuracy. This shows a trade-off between model efficiency and performance.
