# 4. Classification where input is a vector, output is categorical (softmax & negative log likelihood loss).

import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(10000, 3)
Y = torch.tensor(2).repeat(10000)

# params
w = torch.randn(3,3).requires_grad_()
b = torch.tensor(0.1).repeat(3).requires_grad_()

lr = 0.1

batch_size = 100

losses = []

def softmax(logits):
    exps = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return exps / exps.sum(dim=1, keepdim=True) + 1e-6

def nll_loss(y, pred_y):
    return -torch.log(pred_y[range(len(y)), y])

for i in range(len(X) // batch_size):
    from_idx, to_idx = i * batch_size, (i * batch_size) + batch_size
    x = X[from_idx : to_idx]
    y = Y[from_idx : to_idx]

    pred_y= softmax(x @ w.T + b)

    loss = nll_loss(y, pred_y).mean(dim=0)
    if i % 10 == 0:
        print('Loss: ', loss.item())

    loss.backward()

    losses.append(loss.item())

    # update in the negative direction of the gradient
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr

    w.grad = None
    b.grad = None

print('Final params')
print('w', w)
print('b', b)

print(losses)

plt.plot(np.arange(len(losses)), losses)

plt.savefig('3b.png')
