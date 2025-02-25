# 6. Now use N feedforward layers (deep learning!)

import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def relu(x):
    return torch.max(x, torch.tensor(0.0))

class FF:
    def __init__(self, activation=relu):
        self.w = torch.randn(3,3).requires_grad_()
        self.b = torch.tensor(0.1).repeat(3).requires_grad_()
        self.act = activation

    def parameters(self):
        return [self.w, self.b]

    def forward(self, x):
        return self.act(x @ self.w.T + self.b)

class MLP:
    def __init__(self, n=1, activation=relu, last_activation=None):
        self.layers = [FF(last_activation if i == n-1 else activation) for i in range(n)]

    def parameters(self):
        params = []
        for p in self.layers:
            params.extend(p.parameters())
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Optimizer:
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

X = torch.randn(10000, 3)
Y = torch.tensor(2).repeat(10000)

lr = 0.1

batch_size = 100

losses = []

def softmax(logits):
    exps = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return exps / exps.sum(dim=1, keepdim=True) + 1e-6

def nll_loss(y, pred_y):
    return -torch.log(pred_y[range(len(y)), y])

model = MLP(2, activation=relu, last_activation=softmax)
opt = Optimizer(model, lr)

for i in range(len(X) // batch_size):
    opt.zero_grad()

    from_idx, to_idx = i * batch_size, (i * batch_size) + batch_size
    x = X[from_idx : to_idx]
    y = Y[from_idx : to_idx]

    pred_y = model.forward(x)

    loss = nll_loss(y, pred_y).mean(dim=0)
    if i % 10 == 0:
        print('Loss: ', loss.item())

    loss.backward()

    losses.append(loss.item())

    opt.step()

print(losses)

plt.plot(np.arange(len(losses)), losses)

plt.savefig('6.png')
