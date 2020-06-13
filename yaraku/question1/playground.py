"""

Examples:
https://stackoverflow.com/questions/60672361/how-to-plot-the-output-of-k-means-clustering-of-word-embedding-using-python
https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html?highlight=tsne
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne/264647#264647
https://towardsdatascience.com/plotting-text-and-image-vectors-using-t-sne-d0e43e55d89
https://github.com/ashutoshsingh25/Plotting-multidimensional-vectors-using-t-SNE/blob/master/TSNE%20Code%20for%20clusring%20image%20and%20text%20vectors%20with%20labels.ipynb
https://blog.datascienceheroes.com/playing-with-dimensions-from-clustering-pca-t-sne-to-carl-sagan/
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


class GloVe:
    def __init__(self, glove_file: str):
        self._word_to_emb = pd.read_table(
            glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    def __getitem__(self, key: str) -> np.ndarray:
        ret = self._word_to_emb.loc[key].to_numpy()
        return ret

    @property
    def wordset(self) -> Set[str]:
        ret = set(self._word_to_emb.index)
        return ret

    def find_nearest(self, words: List[str], k: int = 5, metric: str = 'cosine'):
        """
        Reference
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh.fit(self._word_to_emb.to_numpy())  # fit the model

        input_ = [self[w] for w in words]
        dists, w_idxs = neigh.kneighbors(
            input_, n_neighbors=k, return_distance=True)
        n_words, _ = dists.shape

        ret = []
        for i in range(n_words):
            tmp = []
            words = self._word_to_emb.iloc[w_idxs[i]].index
            for w, d in zip(words, dists[i]):
                tmp.append((w, d))
            ret.append(tmp)
        return ret

    def get_emb_df(self, words: List[str]):
        ret = self._word_to_emb.loc[words]
        return ret


class WordSample:
    def __init__(self, must_include_wordset: Iterable[str], n_samples=10000):
        with open("./words_alpha.txt", "r") as f:
            self._words = f.read().splitlines()

        print(f"words_alpha.txt has {len(self._words)} words.")
        avaliable = set(self._words).intersection(must_include_wordset)
        print("# of words intersect with `must_include_wordset`: ", len(avaliable))

        # select n_samples data
        self._selected = np.random.choice(
            list(avaliable), size=n_samples, replace=False)
        assert len(set(self._selected)) == n_samples

    @property
    def words(self) -> List[str]:
        return self._selected




if __name__ == "__main__":
    # set random seed
    np.random.seed(4200)

    glove = GloVe("glove.6B.50d.txt")
    words = WordSample(must_include_wordset=glove.wordset, n_samples=100).words

    word_vectors = glove.get_emb_df(words).to_numpy()
    print('Total words:', len(words),
          '\tWord Embedding shapes:', word_vectors.shape)

    metric = "cosine"
    n_neighbors = 100
    perplexity = 30
    n_iter = 1000
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
