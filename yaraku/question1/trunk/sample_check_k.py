"""
An example to show how to check k

Steps
------
1. sample data from the 10000 words
2. business as usual
3. we can see the trend of the sse is the same


Outputs
---------
Loaded the file: glove/glove.6B.50d.txt
Number of words: 400000
./words_alpha.txt has 370099 words.
# of words intersect with `incl_words`:  100776
# of samples: 10000
# of samples for determine k: 3000
n_clusters: 10; inertia: 2318.267822265625; time_used: 1.1969225406646729
n_clusters: 15; inertia: 2237.96728515625; time_used: 1.1593027114868164
n_clusters: 20; inertia: 2180.607421875; time_used: 1.293792963027954
n_clusters: 25; inertia: 2135.764404296875; time_used: 2.0745174884796143
n_clusters: 30; inertia: 2098.356689453125; time_used: 2.164062261581421
n_clusters: 35; inertia: 2066.714599609375; time_used: 2.3054685592651367
n_clusters: 40; inertia: 2038.9697265625; time_used: 2.6846113204956055
n_clusters: 45; inertia: 2017.0618896484375; time_used: 2.2728190422058105
n_clusters: 50; inertia: 1994.0457763671875; time_used: 2.676485300064087
n_clusters: 55; inertia: 1972.0341796875; time_used: 2.7943241596221924
n_clusters: 60; inertia: 1953.3951416015625; time_used: 2.761678457260132
30
2098.356689453125
n_clusters: 65; inertia: 1937.23291015625; time_used: 3.140117883682251
n_clusters: 70; inertia: 1918.1201171875; time_used: 3.6707918643951416
n_clusters: 75; inertia: 1900.1141357421875; time_used: 11.958739995956421
n_clusters: 80; inertia: 1884.15673828125; time_used: 13.850735187530518
n_clusters: 85; inertia: 1873.8721923828125; time_used: 12.359323740005493
n_clusters: 90; inertia: 1861.4041748046875; time_used: 11.179735898971558
n_clusters: 95; inertia: 1846.4525146484375; time_used: 11.910082578659058
n_clusters: 100; inertia: 1835.3214111328125; time_used: 12.430966138839722
n_clusters: 105; inertia: 1819.98291015625; time_used: 11.408290147781372
n_clusters: 110; inertia: 1810.1473388671875; time_used: 12.249155521392822
40
2038.9697265625

"""
from utils.glove import GloVe
from utils.word_sample import WordSample
import numpy as np
from sklearn.cluster import KMeans
import time
from kneed import KneeLocator


def kmean_model_factory(n_clusters):
    ret = KMeans(init="k-means++", n_clusters=n_clusters,
                 random_state=0, verbose=0)
    return ret


if __name__ == "__main__":
    sampling_ratio = 0.1
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

    max_k = min(n_samples_for_det_k, 500)
    list_k = list(range(10, max_k+1, 5))
    sse = dict()
    knee = {}
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
            kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing', online=False, interp_method="interp1d")
            print("knee.x:", kneedle.knee)
            print("knee.y:", kneedle.knee_y)

    x = sorted(sse.keys())
    y = [sse[k] for k in x]
    kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing', online=False, interp_method="interp1d")
    print("knee.x:", kneedle.knee)
    print("knee.y:", kneedle.knee_y)

