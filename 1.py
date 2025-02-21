# Linear regression where input & output are each a single scalar, with stochastic gradient descent (minibatches),
# using numpy. (The trickiest part of this one is implementing backprop by hand.)

import numpy as np

np.random.seed(42)

X = np.random.randn(10000)
Y = X * 1.5

# params
w = 0.1
b = 0.1

lr = 0.1

batch_size = 100

for i in range(len(X) // batch_size):
    from_idx, to_idx = i * batch_size, (i * batch_size) + batch_size

    x = X[from_idx : to_idx]
    y = Y[from_idx : to_idx]

    pred_y = x * w + b

    # L2 loss, mean sq err
    loss = np.sum(np.square(y - pred_y)) / batch_size
    if i % 10 == 0:
        print('Loss: ', loss.item())

    dl_dw = (-2 * np.sum(x * (y - pred_y))) / batch_size
    dl_db = (-2 * np.sum(y - pred_y)) / batch_size

    # update in the negative direction of the gradient
    w -= dl_dw * lr
    b -= dl_db * lr

print('Final params')
print('w', w)
print('b', b)
