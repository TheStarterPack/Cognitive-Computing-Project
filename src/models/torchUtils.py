import torch
import jsonpickle

from torch.utils import data

EMBEDDING_PATH = 'embeddings/'


def data_loader_from_numpy(centers, contexts, batch_size=32,
                           shuffle=True) -> data.DataLoader:
    dataset = data.TensorDataset(torch.from_numpy(centers), torch.from_numpy(contexts))
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def data_set_from_numpy(centers, contexts) -> data.TensorDataset:
    return data.TensorDataset(torch.from_numpy(centers), torch.from_numpy(contexts))

def data_loader_from_x_y(x, y, batch_size=32,
                         shuffle=True) -> data.DataLoader:
    dataset = data.TensorDataset(x, y)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dummy_loader(vocab_size: int, embedding_size: int, context_length: int,
                     batch_size: int = 32) -> data.DataLoader:
    CE = torch.randint(0, vocab_size, size=(embedding_size,))
    CO = torch.randint(0, vocab_size, size=(embedding_size, context_length))
    dataset = data.TensorDataset(CE, CO)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def read_embeddings_dict(approach_name: str) -> dict:
    with open(f'{EMBEDDING_PATH}{approach_name}.json') as file:
        json_string = file.read()
        return jsonpickle.decode(json_string, keys=True)


def write_embeddings_to_file(model,
                             action_to_id: dict, approach_name) -> None:
    id_to_action = {v: k for k, v in action_to_id.items()}
    actions_to_idx_and_embedding = {}
    for idx in range(len(model.contexts)):
        action = id_to_action[idx]
        actions_to_idx_and_embedding[action] = (idx, model.idx_to_center_vec(idx))

    json_string = jsonpickle.encode(actions_to_idx_and_embedding, keys=True)

    with open(f'{EMBEDDING_PATH}{approach_name}.json', 'w') as file:
        file.write(json_string)
