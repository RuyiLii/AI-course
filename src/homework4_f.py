import torch
from torch import nn
from torch import optim
import requests
import pickle
import gzip
import math
from matplotlib import pyplot
import torch.nn.functional as F
from pathlib import Path
import numpy as np

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

        self.weights1 = nn.Parameter(torch.randn(10,20))
        self.weights2 = nn.Parameter(torch.randn(20,10))
        self.bias1 = nn.Parameter(torch.zeros(20))
        self.bias2 = nn.Parameter(torch.zeros(10))


    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        xb = xb.view(1000,10)
        xb = F.relu(xb@self.weights1 + self.bias1)
        xb = xb@self.weights2 + self.bias2
        return log_softmax(xb.view(-1, xb.size(1)))

lr = 0.15
    
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def get_model():
    model = Mnist_CNN()
    return model, optim.SGD(model.parameters(), lr=lr)
epochs = 50
model, opt = get_model()

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
        print(epoch,accuracy(model(xb), yb))

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
bs = 1000
xb = x_train[0:bs] 
yb = y_train[0:bs] 
preds = model(xb)
loss_func = nll
fit()
