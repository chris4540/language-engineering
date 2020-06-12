import numpy as np
import pandas as pd
import csv
from sklearn.neighbors import NearestNeighbors
from typing import List, Set, Iterable


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
    # np.random.seed(45)

    glove = GloVe("glove.42B.300d.txt")
    word_sample = WordSample(must_include_wordset=glove.wordset, n_samples=100)
    print(word_sample.words)
    # print(word_sample.words)

    # vec = glove["hello"]
    # print(glove.find_nearest(["hello", "world"]))
