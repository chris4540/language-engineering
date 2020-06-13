from utils.glove import GloVe
from utils.word_sample import WordSample
from sklearn.cluster import KMeans
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    glove = GloVe("glove.6B.50d.txt")
    words = WordSample(
        "./words_alpha.txt", incl_words=glove.wordset, n_samples=10000).words

    emb_vecs: Dict[str, np.ndarray] = glove.get_emb_vecs_of(words)

    # build the data-matrix with shape = (n_samples, emb_dims)
    X = np.array([emb_vecs[w] for w in words])

    # sse: sum of squared distance
    sse = []
    list_k = [2, 5, 10, 20, 50, 100, 200, 300]
    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(X)
        print(f"n_clusters: {k}; inertia: {km.inertia_}")
        sse.append(km.inertia_)

    # Plot sse against k
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.savefig("sse.pdf")
