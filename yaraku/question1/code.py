import numpy as np
import pandas as pd
import csv


class GloVe:
    def __init__(self, glove_file: str):
        self._word_to_emb = pd.read_table(
            glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    def __getitem__(self, key: str) -> np.ndarray:
        ret = self._word_to_emb.loc[key].to_numpy()
        return ret

    def find_closest_word(self, word_vec):
        pass


if __name__ == "__main__":
    glove = GloVe("glove.6B.50d.txt")
    vec = glove["hello"]
    print(type(vec))
    print(vec)


# def vec(w):
#     ret = words.loc[w].to_list()
#     return ret


# def find_N_closest_word(v, N, words):
#     ret = []
#     for w in range(N):
#         diff = words.as_matrix() - v
#         delta = np.sum(diff * diff, axis=1)
#         i = np.argmin(delta)
#         ret.append(words.iloc[i].name)
#         words = words.drop(words.iloc[i].name, axis=0)

#     return ret

# print(vec('hello'))    # this will print same as print (model['hello'])  before


# print(find_N_closest_word(vec('table'), 10, words))
