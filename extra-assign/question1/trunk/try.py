
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
        input_ = [self[w] for w in words]
        ret = self.find_nearest_by_vec(input_, k=k, metric=metric)
        return ret

    def find_nearest_by_vec(self, vec: list, k=5, metric='cosine'):
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh.fit(self._word_to_emb.to_numpy())  # fit the model
        dists, w_idxs = neigh.kneighbors(
            vec, n_neighbors=k, return_distance=True)
        n_words, _ = dists.shape

        ret = []
        for i in range(n_words):
            tmp = []
            words = self._word_to_emb.iloc[w_idxs[i]].index
            for w, d in zip(words, dists[i]):
                tmp.append((w, d))
            ret.append(tmp)
        return ret

    def get_sub_embedding_df(self, words: List[str]):
        ret = self._word_to_emb.loc[words]
        return ret
