import torch
from torch.utils import data


def data_loader_from_numpy(centers, contexts, batch_size=32,
                           shuffle=True) -> data.DataLoader:
    dataset = data.TensorDataset(torch.from_numpy(centers), torch.from_numpy(contexts))
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dummy_loader(vocab_size: int, embedding_size: int, context_length: int,
                     batch_size: int = 32) -> data.DataLoader:
    CE = torch.randint(0, vocab_size, size=(embedding_size,))
    CO = torch.randint(0, vocab_size, size=(embedding_size, context_length))
    dataset = data.TensorDataset(CE, CO)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
