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
