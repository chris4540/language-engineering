"""

Examples:
https://stackoverflow.com/questions/60672361/how-to-plot-the-output-of-k-means-clustering-of-word-embedding-using-python
https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html?highlight=tsne
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne/264647#264647
https://towardsdatascience.com/plotting-text-and-image-vectors-using-t-sne-d0e43e55d89
https://github.com/ashutoshsingh25/Plotting-multidimensional-vectors-using-t-SNE/blob/master/TSNE%20Code%20for%20clusring%20image%20and%20text%20vectors%20with%20labels.ipynb
https://blog.datascienceheroes.com/playing-with-dimensions-from-clustering-pca-t-sne-to-carl-sagan/
http://xplordat.com/2018/12/14/want-to-cluster-text-try-custom-word-embeddings/
https://stackoverflow.com/questions/46409846/using-k-means-with-cosine-similarity-python
https://stats.stackexchange.com/questions/120350/k-means-on-cosine-similarities-vs-euclidean-distance-lsa
https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
https://github.com/arvkevi/kneed
"""
import numpy as np
import pandas as pd
import csv
from sklearn.neighbors import NearestNeighbors
from typing import List, Set, Iterable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import time
from utils.glove import GloVe


def cosine_sim(v1, v2):
    ret = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return ret

if __name__ == "__main__":
    # set random seed
    # np.random.seed(4200)

    glove = GloVe("glove.6B.300d.txt")
    vec = glove["king"] - glove["man"] + glove["woman"]
    ret = glove.find_nearest_by_vecs([vec])
    print(ret)

    words = WordSample(must_include_wordset=glove.wordset, n_samples=100).words

    # word_vectors = glove.get_emb_df(words).to_numpy()
    # print('Total words:', len(words),
    #       '\tWord Embedding shapes:', word_vectors.shape)

    # metric = "cosine"
    # n_neighbors = 100
    # perplexity = 30
    # n_iter = 1000
    # transformer = make_pipeline(
    #     AnnoyTransformer(n_neighbors=n_neighbors, metric=metric),
    #     TSNE(metric='precomputed', perplexity=perplexity,
    #          method="barnes_hut", random_state=42, n_iter=n_iter))

    # start = time.time()
    # Xt = transformer.fit_transform(X)
    # duration = time.time() - start
    # print(duration)

    # fig, ax = plt.subplot()
    # # ax.set_title(transformer_name + '\non ' + dataset_name)
    # ax.scatter(Xt[:, 0], Xt[:, 1], c=y.astype(np.int32),
    #                 alpha=0.2, cmap=plt.cm.viridis)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.axis('tight')
    # i_ax += 1

    # fig.tight_layout()
    # plt.show()
