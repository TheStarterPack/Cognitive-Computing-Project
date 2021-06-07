import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.models.word2Vec import CustomWord2Vec


def visualize_model_pca(model: CustomWord2Vec, idx_to_action, N=8):
    pca = PCA(n_components=3)
    embeddings = normalize(model.get_averaged_embeddings().detach().numpy())

    array = pca.fit_transform(embeddings)
    first_N = array[:N]

    labels = []
    for i in range(N):
        action = idx_to_action(i)
        labels.append(action.action)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = first_N[:, 0]
    y = first_N[:, 1]
    z = first_N[:, 2]

    for i in range(len(x)):  # plot each point + it's index as text above
        ax.scatter(x[i], y[i], z[i], color='b')
        ax.text(x[i], y[i], z[i], labels[i], size=8, zorder=1,
                color='k')

    plt.show()
