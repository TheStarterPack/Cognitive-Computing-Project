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

EMBEDDING_SIZE = 150
BATCH_SIZE = 64
CONTEXT_LENGTH = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20

TEST_TRAIN_SPLIT = 0.2


class OneLayerNN(nn.Module):
    def __init__(self, context_length: int, embedding_size: int, units: int, num_classes: int):
        super(OneLayerNN, self).__init__()
        self.hidden = nn.Linear(embedding_size * 2 * context_length, units, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.output = nn.Linear(units, num_classes, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.relu(x)
        return x


class LstmNN(nn.Module):

    def __init__(self, context_length: int, embedding_size: int, units: int, num_classes: int,
                 num_layers: int = 1):
        super(LstmNN, self).__init__()

        self.hidden_size = 50
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            bias=True)
        self.bm1 = nn.BatchNorm1d(num_features=self.hidden_size)
        self.output = nn.Linear(self.hidden_size, num_classes, bias=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bm1(x)
        x = nn.ReLU() (x)
        x = self.output(x)
        return x


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


def train(epochs: int, train_loader, model, learning_rate):
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

        print(f'Train Epoch {epoch} Acc: {check_accuracy(train_loader, model)}')
        print(f'Test Epoch {epoch} Acc: {check_accuracy(test_loader, model)}')


def get_prediction_data(approach_name: str, context_size: int, test_train_split: float = 0.2):
    global target_to_id

    action_to_embedding = read_embeddings_dict(approach_name)
    parser = ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    (contexts, centers) = generate_contexts(action_sequences, context_size)

    action_to_occurence = parser.get_str_action_to_occurence()
    target_set = set(action.action for action in chain.from_iterable(action_sequences)
                     if action_to_occurence[action.action] > 500)
    target_to_id = {k: v for k, v in zip(target_set, range(len(target_set)))}
    target_to_id['other'] = len(target_to_id)

    input_data = []
    labels = []
    for i in range(len(contexts)):
        context = contexts[i]
        center = centers[i][0]

        input_data.append(torch.stack([action_to_embedding[action][1] for action in context]))

        if action_to_occurence[center.action] > 500:
            labels.append(target_to_id[center.action])
        else:
            labels.append(target_to_id['other'])

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
    train_loader, test_loader = get_prediction_data('action_target_embedding', CONTEXT_LENGTH)
    num_classes = len(target_to_id)
    print(f'Number of classes: {num_classes}')
    model = LstmNN(CONTEXT_LENGTH, EMBEDDING_SIZE, 20, num_classes)
    model = model.to(DEVICE)

    # test of output dimensions
    x = torch.randn(BATCH_SIZE, CONTEXT_LENGTH * 2, EMBEDDING_SIZE)
    y = model(x)
    assert y.shape == torch.Size([BATCH_SIZE, num_classes])

    train(EPOCHS, train_loader, model, learning_rate=0.005)
    print(check_accuracy(test_loader, model))
    print(check_accuracy(train_loader, model))

    '''
    from sklearn.metrics import accuracy_score

    y_pred = model(np.concatenate([x for x, y in train_loader]))

    accuracy = accuracy_score(np.concatenate([y for x, y in train_loader], axis=0),
                              np.argmax(y_pred, axis=1))
    print(accuracy)
    '''
