# 6. Now use N feedforward layers (deep learning!)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

batch_size = 64

log_every_n_batches = 100

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('./mnist', train=False, transform=transform)

loader_kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(dataset1, **{**loader_kwargs, 'batch_size': batch_size, 'shuffle': True})
test_loader = torch.utils.data.DataLoader(dataset2, **{**loader_kwargs, 'batch_size': batch_size, 'shuffle': False})

def relu(x):
    return torch.max(x, torch.tensor(0.0))

class FF:
    def __init__(self, from_dim, to_dim, activation=relu):
        self.w = torch.randn(from_dim, to_dim).requires_grad_()
        self.b = torch.tensor(0.1).repeat(to_dim).requires_grad_()
        self.act = activation

    def parameters(self):
        return [self.w, self.b]

    def forward(self, x):
        return self.act(x @ self.w + self.b)

class MLP:
    def __init__(self, sizes, activation=relu, last_activation=None):
        self.layers = []
        for idx, (from_size, to_size) in enumerate(zip(sizes, sizes[1:], strict=False)):
            last_one = idx == len(sizes) - 2
            self.layers.append(FF(from_size, to_size, last_activation if last_one else activation))

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

lr = 0.1

train_losses = []
eval_losses = []

def softmax(logits):
    exps = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return exps / exps.sum(dim=1, keepdim=True) + 1e-6

def nll_loss(y, pred_y):
    return -torch.log(pred_y[range(len(y)), y])

model = MLP([784, 64, 10], activation=relu, last_activation=softmax)
opt = Optimizer(model, lr)

def train_epoch():
    for idx, batch in enumerate(train_loader):
        opt.zero_grad()

        x, y = batch

        pred_y = model.forward(x.view(x.shape[0], -1))

        loss = nll_loss(y, pred_y).mean(dim=0)

        if idx % log_every_n_batches == 0:
            print('Loss: ', loss.item())
            train_losses.append(loss.item())

        loss.backward()

        opt.step()

@torch.no_grad()
def eval():
    batch_losses = []
    for batch in test_loader:
        x, y = batch

        pred_y = model.forward(x.view(x.shape[0], -1))

        loss = nll_loss(y, pred_y).mean(dim=0)

        batch_losses.append(loss.item())

    eval_loss = np.mean(batch_losses).item()
    print('Eval loss: ', eval_loss)
    eval_losses.append(eval_loss)

epochs = 10

for epoch in range(epochs):
    print('Epoch ', epoch+1)
    train_epoch()
    eval()

logged_train_losses_per_epoch = len(train_losses) / log_every_n_batches / epochs

plt.figure(figsize=(10, 6))

plt.plot([i * log_every_n_batches for i in range(len(train_losses))], train_losses, label='Train loss')
steps_per_epoch = len(train_loader)
plt.plot([(ep+1) * steps_per_epoch for ep in range(epochs)], eval_losses, label='Eval loss')

plt.legend(loc="upper right")

plt.xlabel("Training steps")
plt.ylabel('Loss')
plt.title('MNIST')

plt.savefig('7_and_8.png')
