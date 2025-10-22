import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.transforms import v2

from torchinfo import summary
import numpy as np
from cnn import convnext_tiny

device_type = 'cuda'
device = torch.device(device_type)

digit_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])

fashion_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])

test_transform = v2.Compose([
    v2.Lambda(lambda x: x.repeat(3, 1, 1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=fashion_transform)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=digit_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    pin_memory_device=device_type,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    pin_memory=True,
    pin_memory_device=device_type
)

def generate_eps(state_dict, standard_deviation):
    eps = standard_deviation * np.random.randn(state_dict.shape)
    return eps

def evolve(state_dict, eps, learning_rate, standard_deviation):
    scores = (scores * eps).sum()
    weights += learning_rate / (weights.numel() * standard_deviation)
    return weights

model = convnext_tiny(num_classes=10).to(device)
state_dict = model.state_dict()

for k, v in state_dict.items():
    print(k, v.shape)

summary(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    losses = []
    for X, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        
        losses.append(loss.item())
    print(np.mean(losses))
        