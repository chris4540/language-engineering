import numpy as np
from typing import List, Set
from sklearn.neighbors import NearestNeighbors


class GloVe:
    def __init__(self, glove_file: str):
        self.embeddings_dict = dict()
        self._words = list()
        with open(glove_file, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
                self._words.append(word)

        print(f"Loaded the file: {glove_file}")

    def __getitem__(self, key: str) -> np.ndarray:
        ret = self.embeddings_dict[key]
        return ret

    @property
    def wordset(self) -> Set[str]:
        ret = set(self.embeddings_dict.keys())
        return ret

    def find_nearest(self, words: List[str], k: int = 5, metric: str = 'cosine'):
        """
        Reference
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        input_ = [self[w] for w in words]
        ret = self.find_nearest_by_vecs(input_, k=k, metric=metric)
        return ret

    def find_nearest_by_vecs(self, vec: List[np.ndarray], k=5, metric='cosine'):
        # gram-matrix
        X = np.array([self[w] for w in self._words])
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh.fit(X)  # fit the model
        dists, w_idxs = neigh.kneighbors(
            vec, n_neighbors=k, return_distance=True)
        n_words, _ = dists.shape

        ret = []
        for i in range(n_words):
            tmp = []
            words = [self._words[idx] for idx in w_idxs[i]]
            for w, d in zip(words, dists[i]):
                tmp.append((w, d))
            ret.append(tmp)
        return ret


def demo():
    glove = GloVe("glove.6B.300d.txt")
    vec = glove["king"] - glove["man"] + glove["woman"]
    ret = glove.find_nearest_by_vecs([vec])
    print(ret)
