# Self-Pruning Neural Network

This project explores a simple idea: what if a neural network could figure out on its own which connections it doesn’t need  while it’s still learning?

Instead of training a full model and pruning it later, I built a network that learns to reduce its own complexity during training. Each weight is paired with a small “gate” that controls how important that connection is. As training progresses, less useful connections are pushed toward zero using an L1 penalty, effectively removing them.

##  What this project does
- Builds a custom linear layer with learnable gates  
- Allows the model to prune itself during training  
- Uses L1 regularization to encourage sparsity  
- Studies how pruning affects accuracy  

## How it works
Every connection in the network has a gate value between 0 and 1. During training, these gates are adjusted along with the weights. By adding a sparsity penalty to the loss, the model is encouraged to turn off unnecessary connections, resulting in a smaller and more efficient network.

##  Results
| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.01   | ~40%     | ~0%      |
| 1.0    | ~38%     | ~5%      |
| 20.0   | ~39%     | ~68%     |

##  How to run
```bash
pip install -r requirements.txt
python train.py
