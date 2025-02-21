# 3. Linear regression where input is a vector.

import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(10000, 3)
Y = (X * 1.5).mean(dim=1)

# params
w = torch.tensor([0.1, 0.1, 0.1], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

lr = 0.1

batch_size = 100

losses = []

for i in range(len(X) // batch_size):
    from_idx, to_idx = i * batch_size, (i * batch_size) + batch_size
    x = X[from_idx : to_idx]
    y = Y[from_idx : to_idx]

    pred_y = x @ w.T + b

    # L2 loss, mean sq err
    loss = torch.sum((y - pred_y)**2) / batch_size
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

plt.savefig('3.png')
