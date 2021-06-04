import torch
import torch.nn as nn
import torch.nn.functional as F

from torchUtils import get_dummy_loader
from torch.optim import Adam, SGD
import torch.optim as optim

action_to_embedding = {}
action_to_id = {}

target_to_id = {}

NUM_CLASSES = 10
EMBEDDING_SIZE = 200
BATCH_SIZE = 64
CONTEXT_LENTGH = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10


class OneLayerNN(nn.Module):
    def __init__(self, context_length: int, embedding_size: int, units: int, num_classes: int):
        super(OneLayerNN, self).__init__()
        self.hidden = nn.Linear(embedding_size, units)
        self.output = nn.Linear(units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


def train(epochs: int, train_loader, model: OneLayerNN, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    citerion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (target, data) in enumerate(train_loader):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            scores = model.forward(data)
            loss = citerion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


if __name__ == '__main__':
    model = OneLayerNN(CONTEXT_LENTGH, EMBEDDING_SIZE, 100, NUM_CLASSES)
    model = model.to(DEVICE)
    x = torch.randn(BATCH_SIZE, EMBEDDING_SIZE)
    y = model(x)
    assert y.shape == torch.Size([BATCH_SIZE, NUM_CLASSES])
    train_loader = get_dummy_loader(1000, EMBEDDING_SIZE, CONTEXT_LENTGH, BATCH_SIZE)
    train(EPOCHS, train_loader, model, learning_rate=0.001)