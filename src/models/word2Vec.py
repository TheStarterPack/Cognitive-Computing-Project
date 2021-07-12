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


class TrainedEmbedding:
    def __init__(self, center, contexts, action):
        self.center,
        self.context,
        self.action,

    def get_averaged_vector(self):
        return (self.centers + self.contexts) / 2

    def get_action_name(self):
        return self.action.action


class CustomWord2Vec(nn.Module):
    def __init__(self, vocab_size: int = 30000, dims: int = 64,
                 name: str = "word2vec") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        self.path = "res/models/" + name + "/"
        self.param_str = f"{dims}dim-"
        self.centers = T.randn(vocab_size, dims, requires_grad=True)
        self.contexts = T.randn(vocab_size, dims, requires_grad=True)
        self.log = {"loss": [], "test_nloss": [], "test_ploss": []}
        self.neg_freq_fac = 1
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.save_every = 5
        self.plot_every = 1

    def training_step(self, batch):
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
        #nloss = 0

        # LOSS
        self.opti.zero_grad()
        loss = ploss + nloss
        loss.backward()
        self.opti.step()

        return loss.item()

    def test(self, loader):
        pls = []
        nls = []
        with T.no_grad():
            for b_idx, batch in enumerate(loader):
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
                #print("NEGATIVES SHAPE", negatives.shape)
                nloss = F.cosine_embedding_loss(centers, negatives, target=-1 * T.ones(c_size))

                # LOSS
                pls.append(ploss.item())
                nls.append(nloss.item())

        
            
        return sum(pls)/len(pls), sum(nls)/len(nls)

    def configure_optimizer(self) -> None:
        self.opti = T.optim.Adam([self.centers, self.contexts])

    def train(self, train_loader: data.DataLoader, test_loader=None, epochs=10, print_every=20) -> None:
        self.centers.to(self.device)
        self.contexts.to(self.device)
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            t = tqdm.tqdm(train_loader)
            for b_idx, batch in enumerate(t):
                for b in batch:
                    b.to(self.device)
                loss = self.training_step(batch)
                epoch_losses.append(loss)

                # PRINT
                if not b_idx % print_every:
                    msg = f"epoch {epoch} loss {round(loss, 3)}"
                    t.set_description(msg)

            self.log["loss"].append(sum(epoch_losses)/len(epoch_losses))

            if test_loader is not None:
                ploss, nloss = self.test(test_loader)
                self.log["test_ploss"].append(ploss)
                self.log["test_nloss"].append(nloss)

            if not epoch % self.plot_every:
                losses_list = ["loss"]+([] if test_loader is None else ["test_ploss", "test_nloss"])
                #print(losses_list)
                self.plot_logs(losses_list)

            if not epoch % self.save_every:
                self.save_model()

    def plot_logs(self, keys, show=False):
        os.makedirs(self.path, exist_ok=True)
        plt.clf()
        for key in keys:
            if self.log[key]:
                #values = self.get_moving_avg(self.log[key], n=30)
                values = self.log[key]
                plt.plot(values, label=key)
                if show:
                    plt.show()
        plt.legend()
        plt.savefig(self.path+self.param_str + "plots.png")

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
        T.save(self.centers.detach(), self.path+self.param_str + "x.pt")
        T.save(self.contexts.detach(), self.path+self.param_str + "y.pt")

    def load_model(self):
        if os.path.exists(self.path+self.param_str+"x.pt"):
            print(f"loading model from {self.path+self.param_str}")
            self.centers = T.load(self.path+self.param_str + "x.pt")
            self.contexts = T.load(self.path+self.param_str + "y.pt")
            self.centers.requires_grad = True
            self.contexts.requires_grad = True
            print("LEAF", self.centers.is_leaf)
            return True
        else:
            print(f"Couldn't find save files for {self.path+self.param_str} -> nothing loaded!")
            return False

    def get_centers(self):
        return self.centers.detach().numpy()

    def get_contexts(self):
        return self.contexts.detach().numpy()


if __name__ == "__main__":
    model = CustomWord2Vec()
    model.configure_optimizer()
    model.train(get_dummy_loader(1000, 1000, 4))
