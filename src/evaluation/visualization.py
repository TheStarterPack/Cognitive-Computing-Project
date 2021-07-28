import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import enum
import xlsxwriter
from src.models.word2Vec import CustomWord2Vec
from nltk.cluster.kmeans import KMeansClusterer
import nltk
import numpy as np
from tqdm import tqdm
import os
import string
import visualize
import sys
from src.parsing.parser import ActionSeqParser
from itertools import takewhile

dir_path = os.path.dirname(os.path.realpath(__file__))


class Distance_function_names(enum.Enum):
    euclidean = 1
    cosine_similarity = 2


def get_distance_function(name: Distance_function_names):
    if name == Distance_function_names.cosine_similarity:
        return lambda embedding1, embedding2: np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    if name == Distance_function_names.euclidean:
        return lambda embedding1, embedding2: distance.euclidean(embedding1, embedding2)
    raise Exception("distance function name not defined")


def do_pca(embeddings, components):
    pca = PCA(n_components=components)
    return pca.fit_transform(embeddings)


def k_means(embeddings, num_classes, repeats):
    kclusterer = KMeansClusterer(num_classes, distance=nltk.cluster.util.cosine_distance, repeats=repeats)
    assigned_embeddings = kclusterer.cluster(embeddings, assign_clusters=True)
    return assigned_embeddings


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


def compute_dist_matrix(embeddings, function_name: Distance_function_names):
    directory = os.path.join(visualize.result_folder, "distance_matrices", str(function_name.name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    dist_matrix_path = os.path.join(directory, "dist_matrix.npy")
    length = len(embeddings)
    matrix = np.zeros((length, length))
    if not os.path.exists(dist_matrix_path):
        print("compute distance matrix...")
        for i in tqdm(range(len(embeddings))):
            for j in range(len(embeddings)):
                if not i == j and not i >= j:
                    matrix[i, j] = get_distance_function(function_name)(embeddings[i], embeddings[j])
        np.save(dist_matrix_path, matrix)
        return matrix
    else:
        print("load distance matrix...")
        with open(dist_matrix_path, 'rb') as f:
            matrix = np.load(f)
            return matrix


def outer_inner_distances(model: CustomWord2Vec, idx_to_action, action_file, target_file,
                          function_name: Distance_function_names):
    embeddings = model.get_embeddings()

    action_targets = get_params(idx_to_action, len(embeddings))
    unique_targets = []
    shortened_action_targets = []
    for target in action_targets:
        new_shortened_target = []
        for target_point in target:
            target_point = ''.join(list(takewhile(lambda x: x != '_', target_point)))
            unique_targets.append(target_point)
            new_shortened_target.append(target_point)
        shortened_action_targets.append(tuple(new_shortened_target))
    unique_targets = list(set(unique_targets))
    print("We have " + str(len(unique_targets)) + " unique targets")

    # get the indexes sorted by action
    targets_indexes = {}
    targets_avg_dist = {}

    for elem in unique_targets:
        targets_indexes[elem] = []
        targets_avg_dist[elem] = [0, 0]
        for i, action_target in enumerate(shortened_action_targets):
            if elem in list(action_target):
                targets_indexes[elem].append(i)

        # compute distance matrix of all embeddings
    dist_matrix = compute_dist_matrix(embeddings, function_name)
    # compute avg distance for inner action distance and outer
    print("compute avg action distance...")
    inner_avg_list = []
    outer_avg_list = []

    # init table data struct
    data = []
    for target in tqdm(unique_targets):
        inner_class_sum = 0
        inner_class_count = 0
        outer_class_sum = 0
        outer_class_count = 0

        # if only one occurence, no distance can be measured
        if len(targets_indexes[target]) == 1:
            print("Action " + target + " occurs only one time.")
            continue
        # compute
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if i in targets_indexes[target] and j in targets_indexes[target]:
                    inner_class_sum += dist_matrix[i, j]
                    inner_class_count += 1
                else:
                    outer_class_sum += dist_matrix[i, j]
                    outer_class_count += 1
        avg_inner_class = np.abs(np.divide(inner_class_sum, inner_class_count))
        avg_outer_class = np.abs(np.divide(outer_class_sum, outer_class_count))
        inner_avg_list.append(avg_inner_class)
        outer_avg_list.append(avg_outer_class)
        data_column = [target, str(len(targets_indexes[target])), str(avg_inner_class), str(avg_outer_class),
                       str(avg_inner_class - avg_outer_class)]
        data.append(data_column)

    row_names = ["target", "occurrences", "inner_dist", "outer_dist", "distance_diff"]
    workbook = xlsxwriter.Workbook(target_file)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', row_names[0], bold)
    worksheet.write('B1', row_names[1], bold)
    worksheet.write('C1', row_names[2], bold)
    worksheet.write('D1', row_names[3], bold)
    worksheet.write('E1', row_names[4], bold)
    for row in range(1, len(data) + 1):
        for col in range(len(row_names)):
            worksheet.write(row, col, data[row - 1][col])
    worksheet.write(len(data) + 1, 2, str(np.average(np.array(inner_avg_list, dtype=float))))
    worksheet.write(len(data) + 1, 3, str(np.average(np.array(outer_avg_list, dtype=float))))
    workbook.close()

    action_names = get_action_names(idx_to_action, len(embeddings))

    # get unique actions and print them
    unique_actions = list(set(action_names))
    print("We have " + str(len(unique_actions)) + " unique actions")

    # get the indexes sorted by action
    action_indexes = {}
    action_avg_dist = {}

    for i, elem in enumerate(action_names):
        if elem not in action_indexes:
            action_indexes[elem] = []
        action_indexes[elem].append(i)
        action_avg_dist[elem] = [0, 0]

    # compute distance matrix of all embeddings
    dist_matrix = compute_dist_matrix(embeddings, function_name)
    # compute avg distance for inner action distance and outer
    print("compute avg target distance...")
    inner_avg_list = []
    outer_avg_list = []

    # init table data struct
    data = []
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
        avg_inner_class = np.abs(np.divide(inner_class_sum, inner_class_count))
        avg_outer_class = np.abs(np.divide(outer_class_sum, outer_class_count))
        inner_avg_list.append(avg_inner_class)
        outer_avg_list.append(avg_outer_class)
        data_column = [action, str(len(action_indexes[action])), str(avg_inner_class), str(avg_outer_class),
                       str(avg_inner_class - avg_outer_class)]
        data.append(data_column)

    row_names = ["action", "occurrences", "inner_dist", "outer_dist", "distance_diff"]
    workbook = xlsxwriter.Workbook(action_file)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write('A1', row_names[0], bold)
    worksheet.write('B1', row_names[1], bold)
    worksheet.write('C1', row_names[2], bold)
    worksheet.write('D1', row_names[3], bold)
    worksheet.write('E1', row_names[4], bold)
    for row in range(1, len(data) + 1):
        for col in range(len(row_names)):
            worksheet.write(row, col, data[row - 1][col])
    worksheet.write(len(data) + 1, 2, str(np.average(np.array(inner_avg_list, dtype=float))))
    worksheet.write(len(data) + 1, 3, str(np.average(np.array(outer_avg_list, dtype=float))))
    workbook.close()


def visualize_model_pca(model: CustomWord2Vec, idx_to_action, n=20):
    embeddings = model.get_embeddings()
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
