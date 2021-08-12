from collections import OrderedDict
from sklearn import cluster
from src.parsing import parser, actionSequence as AS
from src.models import word2Vec
import argparse
from src.models.torchUtils import data_loader_from_numpy
from sklearn.cluster import KMeans
from src.models.torchUtils import data_set_from_numpy
from src.models.torchUtils import write_embeddings_to_file
from torch.utils.data import random_split, DataLoader
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import torch as T
from matplotlib import pyplot as plt
import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
import nltk


if __name__ == '__main__':
    # SETUP ARGUMENT PARSER
    argpar = argparse.ArgumentParser()
    argpar.add_argument("--epochs", type=int, default=30)
    argpar.add_argument("--ncluster", type=int, default=3)
    argpar.add_argument("--dims", type=int, default=64)
    argpar.add_argument("-noload", action="store_true")
    argpar.add_argument("-train", action="store_true")
    argpar.add_argument("-fused", action="store_true")
    argpar.add_argument("-pca", action="store_true")
    argpar.add_argument("-actions", action="store_true")
    args = argpar.parse_args()
    NAME = f"kmeans-{'fused-' if args.fused else ''}{args.ncluster}-clusters-3dim-{'pca' if args.pca else ''}"

    # SETUP PARSER
    parser = parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    action_to_id = parser.get_action_to_id_dict()

    # SETUP MODEL
    vocab_size = len(action_to_id)
    print(f"vocab size: {vocab_size}")
    model = word2Vec.CustomWord2Vec(vocab_size=vocab_size, dims=args.dims)
    loaded_model_flag = False
    if not args.noload:
        loaded_model_flag = model.load_model()
    model.configure_optimizer()

    # SETUP DATA
    contexts, centers = AS.generate_contexts(action_sequences)
    np_contexts = AS.actions_to_tokenized_np_arrays(contexts, action_to_id)
    np_centers = AS.actions_to_tokenized_np_arrays(centers, action_to_id)
    dataset = data_set_from_numpy(np_centers.squeeze(), np_contexts)
    train_counts = int(0.9*len(dataset))
    trainset, testset = random_split(dataset, (train_counts, len(dataset)-train_counts))
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)

    # TRAINING
    if not loaded_model_flag or args.train:
        print(f"Start of Training for {args.epochs} epochs")
        model.train(train_loader, test_loader=test_loader, epochs=args.epochs)

    #write_embeddings_to_file(model, action_to_id, approach_name='action_target_embedding')

    # TESTING
    #idx_to_action = lambda idx: list(action_to_id.keys())[list(action_to_id.values()).index(idx)]
    idx_to_action = {v:k for k,v in action_to_id.items()}

    X = model.centers.detach().numpy()
    if args.fused:
        X += model.contexts.detach().numpy()
    if args.pca:
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
    print("SHAPE", X.shape)

    #clusterer = KMeans(n_clusters=args.ncluster).fit(X)
    clusterer = KMeansClusterer(args.ncluster, distance=nltk.cluster.util.cosine_distance, repeats=10)
    labels = clusterer.cluster(X, assign_clusters=True)
    #print("assigned embeds:", assigned_embeddings)
    #clusterer = DBSCAN(metric="cosine").fit(X)
    #cluster_centers = pca.inverse_transform(clusterer.cluster_centers_)

    writefile = open(f"results/{NAME}.txt", "w")
    for c_idx in range(args.ncluster):
        #print(c_idx, [str(idx_to_action[idx]) for idx in model.get_most_similar_idxs(vec=T.from_numpy(center))], file=writefile)
        full_actions = [str(idx_to_action[idx]) for idx in range(len(X)) if labels[idx]==c_idx]
        if args.actions:
            action_beginnings = [item.split(",")[0] for item in full_actions]
        else:
            action_beginnings = [item.split("\'")[1] if len(item.split("\'"))>1 else "None" for item in full_actions]
        action_counts = OrderedDict()
        for action in action_beginnings:
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 0
        top_action_beginning_idxs = np.argsort(np.array(list(action_counts.values())))[-10:]
        action_beginning_names = list(action_counts.keys())
        top_action_beginnings =[action_beginning_names[idx] for idx in top_action_beginning_idxs]
        print(c_idx, top_action_beginnings, file=writefile)

    #labels = clusterer.labels_
    if args.pca:
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2], c=labels)
        ax.set_title(NAME)
        plt.savefig(f"results/{NAME}.png")
        plt.show()