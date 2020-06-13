import numpy as np


def cosine_sim(v1, v2):
    ret = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return ret
