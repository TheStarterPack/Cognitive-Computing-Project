import os

from src.models.torchUtils import get_dummy_loader

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from matplotlib import pyplot as plt
import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))


class CustomWord2Vec(nn.Module):
    def __init__(self, vocab_size: int = 30000, dims: int = 150,
                 name: str = "default-word2vec") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        self.path = "res/models/" + name + "/"
        self.centers = T.randn(vocab_size, dims, requires_grad=True)
        self.contexts = T.randn(vocab_size, dims, requires_grad=True)
        self.log = {"loss": []}
        self.neg_freq_fac = 1
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.save_every = 5
        self.plot_every = 1

    def training_step(self, batch: [T.Tensor]):
        # BATCH
        center_idxs, context_idxs = batch
        n_contexts = context_idxs.shape[1]
        centers = self.centers[center_idxs]
        centers = centers.repeat_interleave(n_contexts, dim=0)
        contexts = self.contexts[context_idxs].flatten(0, 1)
        c_size = contexts.shape[0]

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

        return loss.item()

    def configure_optimizer(self) -> None:
        self.opti = T.optim.Adam([self.centers, self.contexts])

    def train(self, train_loader: data.DataLoader, epochs=10, print_every=20) -> None:
        self.centers.to(self.device)
        self.contexts.to(self.device)
        for epoch in range(1, epochs + 1):
            t = tqdm.tqdm(train_loader)
            for b_idx, batch in enumerate(t):
                for b in batch:
                    b.to(self.device)
                loss = self.training_step(batch)

                # PRINT
                if not b_idx % print_every:
                    msg = f"epoch {epoch} loss {round(loss, 3)}"
                    t.set_description(msg)

            if not epoch % self.plot_every:
                self.plot_logs(["loss"])
            if not epoch % self.save_every:
                self.save_model()

    def plot_logs(self, keys, show=False):
        os.makedirs(self.path, exist_ok=True)
        for key in keys:
            if self.log[key]:
                plt.clf()
                values = self.get_moving_avg(self.log[key], n=30)
                plt.plot(values, label=key)
                if show:
                    plt.show()
                plt.savefig(self.path + "plots.png")

    def get_moving_avg(self, x, n=10):
        cumsum = np.cumsum(x)
        return (cumsum[n:] - cumsum[:-n]) / n

    def idx_to_center_vec(self, idx):
        return self.centers[idx]

    def idx_to_context_vec(self, idx):
        return self.contexts[idx]

    def get_most_similar_idxs(self, idx=None, vec=None, centers=False, top_n=10):
        assert idx is not None or vec is not None
        contained = int(vec is None)
        vectors = self.centers if centers else self.contexts
        if contained:
            vec = vectors[idx]

        norm = T.norm(vec)
        all_norms = T.norm(vectors, dim=1)
        sims = vec.matmul(vectors.T) / (norm * all_norms)
        top = T.topk(sims, top_n + contained)[1][contained:10 + contained]
        return top.tolist()

    def save_model(self):
        os.makedirs(self.path, exist_ok=True)
        print(f"saving model to {self.path}")
        T.save(self.centers.detach(), self.path + "x.pt")
        T.save(self.contexts.detach(), self.path + "y.pt")

    def load_model(self):
        if os.path.exists(self.path+"x.pt"):
            print(f"loading model from {self.path}")
            self.centers = T.load(self.path + "x.pt")
            self.contexts = T.load(self.path + "y.pt")
            self.centers.requires_grad = True
            self.contexts.requires_grad = True
            #print("LEAF", self.centers.is_leaf)
            return True
        else:
            print(f"Couldn't find save files in path {self.path} -> nothing loaded!")
            return False

    def get_averaged_embeddings(self):
        return (self.centers + self.contexts) / 2


if __name__ == "__main__":
    model = CustomWord2Vec()
    model.configure_optimizer()
    model.train(get_dummy_loader(1000, 1000, 4))
