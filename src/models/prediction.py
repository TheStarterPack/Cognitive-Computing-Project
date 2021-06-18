import random
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import ReLU
from src.parsing.actionSequence import generate_contexts
from src.parsing.parser import ActionSeqParser
from torchUtils import read_embeddings_dict, data_loader_from_x_y

target_to_id = {}

EMBEDDING_SIZE = 64
BATCH_SIZE = 32
CONTEXT_LENTGH = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10

TEST_TRAIN_SPLIT = 0.1


class LstmNN(nn.Module):

    def __init__(self, input_size=76, hidden_size=150, num_layers=2, num_classes=76):
        super(LstmNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.8)
        self.batchNorm = nn.BatchNorm1d(num_features=hidden_size)
        self.relu = ReLU()
    def forward(self, x):
        x, (ht, ct) = self.lstm(x)
        x = self.batchNorm(ht[-1])
        x = self.linear(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


def train(model: LstmNN, train_set):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCHS):
        for batch_idx, (inputs, targets) in enumerate(train_set):
            model.zero_grad()

            output = model(inputs)
            loss = loss_function(output, targets)
            loss.backward()
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
        input_data.append(
            torch.stack([action_to_embedding[action][1].clone().detach() for action in context]))
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
    model = LstmNN(input_size=64, num_classes=num_classes)
    model = model.to(DEVICE)

    train(model, train_loader)
    print(check_accuracy(test_loader, model))
    print(check_accuracy(train_loader, model))
