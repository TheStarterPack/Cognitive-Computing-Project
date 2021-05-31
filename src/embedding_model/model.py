import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from src.model.Action import Action
from src.model.ActionSequence import ActionSequence
from src.parsing import Parser


class CustomWord2Vec(nn.Module):
    def __init__(self, vocab_size: int = 30000, dims: int = 128,
                 path: str = "default-model/") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        self.path = path
        self.centers = T.randn(vocab_size, dims, requires_grad=True)
        self.contexts = T.randn(vocab_size, dims, requires_grad=True)
        self.log = {"loss": []}
        self.neg_freq_fac = 1
        self.device = "cuda" if T.cuda.is_available() else "cpu"

    def training_step(self, batch):
        # BATCH
        center_idxs, context_idxs = batch
        n_contexts = context_idxs.shape[1]
        centers = self.centers[center_idxs]
        centers = centers.repeat_interleave(n_contexts, dim=0)
        contexts = self.contexts[context_idxs].flatten(0, 1)
        c_size = contexts.shape[0]
        # print(centers.shape, contexts.shape)

        # POSITIVE
        ploss = F.cosine_embedding_loss(centers, contexts, target=T.ones(c_size))

        # NEGATIVE
        negatives = self.contexts[T.randint(self.vocab_size, size=(self.neg_freq_fac * c_size,))]
        nloss = F.cosine_embedding_loss(centers, negatives, target=-1 * T.ones(c_size))

        # LOSS
        self.opti.zero_grad()
        loss = ploss + nloss
        self.log["loss"].append(loss.item())
        loss.backward()
        self.opti.step()

        return loss.item(),

    def configure_optimizer(self):
        self.opti = T.optim.Adam([self.centers, self.contexts])

    def get_dummy_loader(self):
        CE = T.randint(0, self.vocab_size, size=(1000,))
        CO = T.randint(0, self.vocab_size, size=(1000, 4))
        dataset = data.TensorDataset(CE, CO)
        return data.DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self, train_loader: data.DataLoader, epochs=10, print_every=20):
        self.centers.to(self.device)
        self.contexts.to(self.device)
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                for b in batch:
                    b.to(self.device)
                ret = self.training_step(batch)

                # PRINT
                if not batch_idx % print_every:
                    print(f"e{epoch} b{batch_idx}:", end="")
                    print(*ret)


def generate_contexts(action_sequences: [ActionSequence], context_length: int = 2):
    assert context_length >= 1

    contexts = []
    centers = []
    for sequence in action_sequences:
        assert isinstance(sequence, ActionSequence)
        for i in range(context_length, len(sequence) - context_length - 1):
            right_context = sequence[i - context_length: i]
            left_context = sequence[i + 1: i + context_length + 1]
            context = right_context + left_context
            center = sequence[i]

            assert isinstance(center, Action)
            assert len(right_context) == len(left_context)
            assert len(context) == context_length * 2
            assert right_context + [center] + left_context == \
                   sequence[i - context_length: i + context_length + 1]

            contexts.append(context)

            centers.append([center])

    return contexts, centers


def actions_to_tokenized_np_arrays(action_lists: [[Action]], action_to_id: dict):
    tokenized_actions = []
    for actions in action_lists:
        np_array = np.fromiter((action_to_id[action] for action in actions), dtype=np.int64)
        tokenized_actions.append(np_array)

    return np.stack(tokenized_actions)


if __name__ == "__main__":
    model = CustomWord2Vec()
    model.configure_optimizer()

    parser = Parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    action_to_id = parser.get_action_to_id_dict()
    contexts, centers = generate_contexts(action_sequences)

    np_contexts = actions_to_tokenized_np_arrays(contexts, action_to_id)
    np_centers = actions_to_tokenized_np_arrays(centers, action_to_id)
