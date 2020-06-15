"""
An example to show how to check k

Steps
------
1. sample data from the 10000 words
2. business as usual
3. we can see the trend of the sse is the same


Outputs
---------
(qgen) chrislin@PC-B82:~/.../yaraku/question1(master)$ python -m trunk.check_k
Loaded the file: glove/glove.6B.50d.txt
Number of words: 400000
./words_alpha.txt has 370099 words.
# of words intersect with `incl_words`:  100776
# of samples: 10000
# of samples for determine k: 10000
max_k: 200
n_clusters: 10; inertia: 7766.49560546875; time_used: 1.5017313957214355
n_clusters: 20; inertia: 7345.90771484375; time_used: 1.753089427947998
n_clusters: 30; inertia: 7075.78662109375; time_used: 3.9235267639160156
n_clusters: 40; inertia: 6891.92822265625; time_used: 4.184268236160278
n_clusters: 50; inertia: 6744.302734375; time_used: 4.357874393463135
n_clusters: 60; inertia: 6628.7275390625; time_used: 4.677428960800171
n_clusters: 70; inertia: 6528.431640625; time_used: 4.981411695480347
n_clusters: 80; inertia: 6447.98193359375; time_used: 27.469621181488037
n_clusters: 90; inertia: 6368.736328125; time_used: 28.1209557056427
n_clusters: 100; inertia: 6310.6123046875; time_used: 27.8488507270813
n_clusters: 110; inertia: 6236.91357421875; time_used: 28.274120807647705
knee.x: 40
knee.y: 6891.92822265625
n_clusters: 120; inertia: 6195.2373046875; time_used: 25.19417691230774
n_clusters: 130; inertia: 6140.236328125; time_used: 29.529817581176758
n_clusters: 140; inertia: 6089.92333984375; time_used: 28.500284433364868
n_clusters: 150; inertia: 6041.06005859375; time_used: 25.215939044952393
n_clusters: 160; inertia: 6003.9365234375; time_used: 25.263726234436035
n_clusters: 170; inertia: 5961.67431640625; time_used: 25.660285234451294
n_clusters: 180; inertia: 5927.01123046875; time_used: 25.229716777801514
n_clusters: 190; inertia: 5891.560546875; time_used: 26.159708738327026
n_clusters: 200; inertia: 5858.35009765625; time_used: 24.740007162094116
knee.x: 60
knee.y: 6628.7275390625
"""
from utils.glove import GloVe
from utils.word_sample import WordSample
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time
from kneed import KneeLocator


def kmean_model_factory(n_clusters):
    # ret = KMeans(init="k-means++", n_clusters=n_clusters,
    #              random_state=0, verbose=0)
    ret = MiniBatchKMeans(
        init='k-means++', n_clusters=n_clusters, batch_size=500,
        n_init=10, max_no_improvement=10, init_size=3*n_clusters,
        random_state=0, verbose=0)
    return ret


if __name__ == "__main__":
    sampling_ratio = 1
    n_samples = 10000
    glove = GloVe("glove/glove.6B.50d.txt")
    words = WordSample(
        "./words_alpha.txt", incl_words=glove.wordset, n_samples=n_samples).words

    n_samples_for_det_k = int(sampling_ratio * n_samples)
    print(f"# of samples: {n_samples}")
    print(f"# of samples for determine k: {n_samples_for_det_k}")
    samples = np.random.RandomState(10).choice(words, size=n_samples_for_det_k, replace=False)

    emb_vecs = glove.get_emb_vecs_of(samples)
    # build the data-matrix with shape = (n_samples, emb_dims)
    X = np.array([emb_vecs[w] for w in samples])
    # normalize it
    length = np.sqrt((X**2).sum(axis=1))[:, None]
    X = X / length

    max_k = 200
    print(f"max_k: {max_k}")
    list_k = list(range(10, max_k+1, 5))
    sse = dict()
    for i, k in enumerate(list_k):
        stime = time.time()
        km = kmean_model_factory(k)
        km.fit(X)
        time_used = time.time() - stime
        print(
            f"n_clusters: {k}; inertia: {km.inertia_}; time_used: {time_used}")
        sse[k] = km.inertia_

        if i > 0 and i % 10 == 0:
            x = sorted(sse.keys())
            y = [sse[k] for k in x]
            kneedle = KneeLocator(x, y, curve='convex', direction='decreasing', interp_method='polynomial')
            print("knee.x:", kneedle.knee)
            print("knee.y:", kneedle.knee_y)

    x = sorted(sse.keys())
    y = [sse[k] for k in x]
    kneedle = KneeLocator(x, y, curve='convex', direction='decreasing', interp_method='polynomial')
    print("knee.x:", kneedle.knee)
    print("knee.y:", kneedle.knee_y)

