from utils.glove import GloVe
from utils.word_sample import WordSample
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
import json


def get_kmean_model(n_clusters):
    # ret = MiniBatchKMeans(
    #     init='k-means++', n_clusters=n_clusters, batch_size=100,
    #     n_init=10, max_no_improvement=10, init_size=3*n_clusters,
    #     random_state=0, max_iter=1000)
    ret = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=0, verbose=0)
    return ret


if __name__ == "__main__":
    glove = GloVe("glove.6B.300d.txt")
    words = WordSample(
        "./words_alpha.txt", incl_words=glove.wordset, n_samples=10000).words

    emb_vecs: Dict[str, np.ndarray] = glove.get_emb_vecs_of(words)

    # build the data-matrix with shape = (n_samples, emb_dims)
    X = np.array([emb_vecs[w] for w in words])
    # normalize it
    length = np.sqrt((X**2).sum(axis=1))[:, None]
    X = X / length

    # sse: sum of squared distance
    sse = dict()
    # list_k = [10, 50, 100, 150, 200, 300, 500, 1000]
    list_k = [1, 2, 5]
    list_k = list(range(5, 300+1, 10))
    for k in list_k:
        stime = time.time()
        km = get_kmean_model(k)
        km.fit(X)
        time_used = time.time() - stime
        print(
            f"n_clusters: {k}; inertia: {km.inertia_}; time_used: {time_used}")
        sse[k] = km.inertia_
    # Plot sse against k
    plt.style.use('fivethirtyeight')
    plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.plot(list_k, [sse[k] for k in list_k], '-o')
    ax.set_xlabel(r'Number of clusters *k*')
    ax.set_ylabel('Sum of squared distance')
    # ---------------------------------------
    plt.tight_layout()
    plt.savefig("elbow.png")

    # save dict
    with open("sse_final.json", "w") as f:
        json.dump(sse, f)
    # ---------------------------------------
    # # focus from 20 to 100
    # list_k = list(range(10, 150, 5))
    # for k in list_k:
    #     if k in sse:
    #         print(f"Skipping the n_clusters: {k}")
    #         continue
    #     stime = time.time()
    #     # km = MiniBatchKMeans(n_clusters=k, init_size=3*k, random_state=30)
    #     # km = KMeans(n_clusters=k, random_state=30)
    #     km = get_kmean_model(k)
    #     km.fit(X)
    #     time_used = time.time() - stime
    #     print(
    #         f"n_clusters: {k}; inertia: {km.inertia_}; time_used: {time_used}")
    #     sse[k] = km.inertia_

    # # Plot sse against k
    # plt.style.use('fivethirtyeight')
    # plt.figure()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(18, 7)
    # ax.plot(list_k, [sse[k] for k in list_k], '-o')
    # ax.set_xlabel(r'Number of clusters *k*')
    # ax.set_ylabel('Sum of squared distance')
    # # ---------------------------------------
    # plt.tight_layout()
    # plt.savefig("elbow_fine.png")
