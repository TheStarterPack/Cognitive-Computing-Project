import random
from itertools import chain
import torch
import torch.nn as nn

from src.parsing.actionSequence import generate_contexts
from src.parsing.parser import ActionSeqParser
import torch.optim as optim
from torchUtils import read_embeddings_dict, data_loader_from_x_y
import torch.nn.functional as F


target_to_id = {}

EMBEDDING_SIZE = 64
BATCH_SIZE = 32
CONTEXT_LENTGH = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5

TEST_TRAIN_SPLIT = 0.001


class OneLayerNN(nn.Module):
    def __init__(self, context_length: int, embedding_size: int, units: int, num_classes: int):
        super(OneLayerNN, self).__init__()
        self.hidden = nn.Linear(embedding_size * 2 * context_length, units)
        self.output = nn.Linear(units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.relu(x)
        x = self.softmax(x)
        return x


def train(epochs: int, train_loader, model: OneLayerNN, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    citerion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (input_data, target) in enumerate(train_loader):
            input_data = input_data.to(device=DEVICE)
            target = target.to(device=DEVICE)
            prediction = model.forward(input_data)
            loss = citerion(prediction, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for input_data, target in loader:
            input_data = input_data.to(device=DEVICE)
            target = target.to(device=DEVICE)

            scores = model(input_data)
            _, predictions = scores.max(1)
            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def get_prediction_data(approach_name: str, context_size: int, test_train_split: float = 0.2):
    global target_to_id
    action_to_embedding = read_embeddings_dict(approach_name)
    parser = ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    (contexts, centers) = generate_contexts(action_sequences, context_size)

    target_set = set(action.action for action in chain.from_iterable(action_sequences))
    target_to_id = {k: v for k, v in zip(target_set, range(len(target_set)))}

    input_data = []
    labels = []
    for i in range(len(contexts)):
        context = contexts[i]
        center = centers[i][0]
        input_data.append(torch.stack([action_to_embedding[action][1] for action in context]))
        labels.append(target_to_id[center.action])

    assert len(input_data) == len(labels)

    c = list(zip(input_data, labels))
    random.shuffle(c)
    input_data, labels = zip(*c)

    idx_split = int(len(input_data) * test_train_split)

    return data_loader_from_x_y(torch.stack(input_data[:idx_split]),
                                torch.LongTensor(labels[:idx_split])), \
           data_loader_from_x_y(torch.stack(input_data[idx_split:]),
                                torch.LongTensor(labels[idx_split:]))


if __name__ == '__main__':
    train_loader, test_loader = get_prediction_data('action_target_embedding', CONTEXT_LENTGH)
    num_classes = len(target_to_id)
    model = OneLayerNN(CONTEXT_LENTGH, EMBEDDING_SIZE, 10, num_classes)
    model = model.to(DEVICE)

    # test of output dimensions
    x = torch.randn(BATCH_SIZE, CONTEXT_LENTGH * 2, EMBEDDING_SIZE)
    y = model(x)
    assert y.shape == torch.Size([BATCH_SIZE, num_classes])

    train(EPOCHS, train_loader, model, learning_rate=0.01)
    print(check_accuracy(train_loader, model))
    print(check_accuracy(test_loader, model))
