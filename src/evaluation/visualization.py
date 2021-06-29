import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from src.models.word2Vec import CustomWord2Vec
from nltk.cluster.kmeans import KMeansClusterer
import nltk
import numpy as np
from tqdm import tqdm
import os
from prettytable import PrettyTable
import dash
import dash_table
from src.parsing.parser import ActionSeqParser

dir_path = os.path.dirname(os.path.realpath(__file__))


def do_pca(embeddings, components):
    pca = PCA(n_components=components)
    return pca.fit_transform(embeddings)


def k_means(embeddings, num_classes, repeats):
    kclusterer = KMeansClusterer(num_classes, distance=nltk.cluster.util.cosine_distance, repeats=repeats)
    assigned_embeddings = kclusterer.cluster(embeddings, assign_clusters=True)
    return assigned_embeddings


def get_avg_embedding(centers, contexts):
    return np.divide(np.add(centers, contexts), 2)


def get_action_names(idx_to_action, embedding_num):
    labels = []
    for i in range(embedding_num):
        action = idx_to_action[i]
        labels.append(action.action)
    return labels


def get_params(idx_to_action, embedding_num):
    labels = []
    for i in range(embedding_num):
        action = idx_to_action[i]
        labels.append(action.targets)
    return labels


def compute_dist_matrix(embeddings):
    length = len(embeddings)
    matrix = np.zeros((length, length))
    if not os.path.exists(os.path.join(dir_path, "dist_matrix.npy")):
        print("compute distance matrix...")
        for i in tqdm(range(len(embeddings))):
            for j in range(len(embeddings)):
                if not i == j and not i >= j:
                    matrix[i, j] = compute_cos_distance(embeddings[i], embeddings[j])
        np.save(os.path.join(dir_path, "dist_matrix.npy"), matrix)
        return matrix
    else:
        print("load distance matrix...")
        with open(os.path.join(dir_path, "dist_matrix.npy"), 'rb') as f:
            matrix = np.load(f)
            return matrix


def compute_cos_distance(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def outer_inner_distances(model: CustomWord2Vec, idx_to_action):
    app = dash.Dash("Result table")
    app.layout = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in ["action_str", "occurences", "inner distance", "outer distance"]],
        data=None,
    )
    app.run_server(debug=True)

    embeddings = get_avg_embedding(model.get_centers(), model.get_contexts())

    action_names = get_action_names(idx_to_action, len(embeddings))
    parser = ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    print(len(action_sequences))
    action_to_id = parser.get_action_to_id_dict()
    # get unique actions and print them
    unique_actions = list(set(action_names))
    print("We have " + str(len(unique_actions)) + " unique actions:")
    for action in unique_actions:
        print(action)

    # get the indexes sorted by action
    action_indexes = {}
    action_avg_dist = {}
    for i, elem in enumerate(action_names):
        if elem not in action_indexes:
            action_indexes[elem] = []
        action_indexes[elem].append(i)
        action_avg_dist[elem] = [0, 0]

    # compute distance matrix of all embeddings
    dist_matrix = compute_dist_matrix(embeddings)
    # compute avg distance for inner action distance and outer
    print("compute avg action distance...")
    smaller_actions = []
    inner_avg_list = []
    outer_avg_list = []
    for action in tqdm(unique_actions):
        inner_class_sum = 0
        inner_class_count = 0
        outer_class_sum = 0
        outer_class_count = 0

        # if only one occurence, no distance can be measured
        if len(action_indexes[action]) == 1:
            print("Action " + action + " occurs only one time.")
            continue
        # compute
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if i in action_indexes[action] and j in action_indexes[action]:
                    inner_class_sum += dist_matrix[i, j]
                    inner_class_count += 1
                else:
                    outer_class_sum += dist_matrix[i, j]
                    outer_class_count += 1
        avg_outer_class = np.abs(np.divide(inner_class_sum, inner_class_count))
        avg_inner_class = np.abs(np.divide(outer_class_sum, outer_class_count))
        inner_avg_list.append(avg_inner_class)
        outer_avg_list.append(avg_outer_class)
        if avg_inner_class < avg_outer_class:
            smaller_actions.append(action)
        print(action + ": (" + str(avg_inner_class) + "),(" + str(avg_outer_class) +
              ") Occurrence: " + str(len(action_indexes[action])) + " times")
    print("From " + str(len(unique_actions)) + "actions " + str(len(smaller_actions)) + "inner averages where smaller than outer")
    for action in smaller_actions:
        print(action)
    print("Avg inner class: " + str(np.average(np.array(inner_avg_list, dtype=float))))
    print("Avg outer class: " + str(np.average(np.array(outer_avg_list, dtype=float))))


def visualize_model_pca(model: CustomWord2Vec, idx_to_action, n=20):
    embeddings = get_avg_embedding(model.get_centers(), model.get_contexts())
    reduced_embeddings = do_pca(embeddings, 3)
    classes = k_means(embeddings, 3, 25)
    print(classes.count(0))
    print(classes.count(1))
    print(classes.count(2))
    action_names = get_action_names(idx_to_action, len(embeddings))

    take_num = 1000
    idx_to_take = np.random.choice(range(len(embeddings)), take_num, replace=False)
    combined_embeddings = []
    for idx in range(len(embeddings)):
        combined_embeddings.append([reduced_embeddings[idx], classes[idx], action_names[idx]])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(combined_embeddings)):  # plot each point + it's index as text above
        x = combined_embeddings[i][0][0]
        y = combined_embeddings[i][0][1]
        z = combined_embeddings[i][0][2]
        if combined_embeddings[i][1] == 0:
            ax.scatter(x, y, z, color='b')
        elif combined_embeddings[i][1] == 1:
            ax.scatter(x, y, z, color='r')
        else:
            ax.scatter(x, y, z, color='g')
        # ax.text(x, y, z, combined_embeddings[i][2], size=8, zorder=1, color='k')

    plt.show()
