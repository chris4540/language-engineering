"""
Use vscode interactive and then convert this file to jupyter
See also: https://code.visualstudio.com/docs/python/jupyter-support-py#_export-a-jupyter-notebook

Usage:
  $ ipython word_clustering_nb.py

Outline:
- [x] sampling words
- [x] load word embeddings
- [x] tidy up
- [x] determine how many clusters
- [x] Do clustering
- [x] Display samples from each clusters
- [x] Display centers
"""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator
import time
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
from typing import List, Dict, Iterable
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from pathlib import Path
import json
# inline matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
# set seaborn style
sns.set_style('darkgrid')


# %%
class GloVe:
    def __init__(self, glove_file: str):
        self.embeddings_dict = dict()
        self._words = list()
        with open(glove_file, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
                self._words.append(word)

                if i % 100000 == 0:
                    print(f"Processed {i} items")

        print(f"Loaded the file: {glove_file}")
        print(f"Number of words: {len(self._words)}")

    def __getitem__(self, key: str) -> np.ndarray:
        ret = self.embeddings_dict[key]
        return ret

    @property
    def words(self) -> List[str]:
        ret = list(self._words)
        return ret

    @words.setter
    def words(self, val: Iterable[str]):
        words = []
        for w in val:
            if w not in self.embeddings_dict:
                continue
            else:
                words.append(w)

        self._words = words

    def find_nearest(self, words: List[str], k: int = 5, metric: str = 'cosine'):
        """
        Reference
        ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        input_ = [self[w] for w in words]
        ret = self.find_nearest_by_vecs(input_, k=k, metric=metric)
        return ret

    def find_nearest_by_vecs(self, vecs: List[np.ndarray], k=5, metric='cosine'):
        # gram-matrix
        X = np.array([self[w] for w in self._words])
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh.fit(X)  # fit the model
        dists, w_idxs = neigh.kneighbors(
            vecs, n_neighbors=k, return_distance=True)
        n_words, _ = dists.shape

        ret = []
        for i in range(n_words):
            tmp = []
            words = [self._words[idx] for idx in w_idxs[i]]
            for w, d in zip(words, dists[i]):
                tmp.append((w, d))
            ret.append(tmp)
        return ret

    def find_nearest_by_vec(self, vec: np.ndarray, k: int = 5, metric: str = 'cosine'):
        rets = self.find_nearest_by_vecs([vec], k=k, metric=metric)
        # return only the first one
        ret = rets[0]
        return ret

    def get_emb_vecs_of(self, words: List[str]) -> Dict[str, np.ndarray]:
        ret = {w: self[w] for w in words}
        return ret


# %%
class WordSampler:
    def __init__(self, file, incl_words: Iterable[str], n_samples=10000, random_state=40):
        with open(file, "r", encoding="utf-8") as f:
            self._words = f.read().splitlines()

        print(f"{file} has {len(self._words)} words.")
        avaliable = set(self._words).intersection(incl_words)
        print("# of words intersect with `incl_words`: ", len(avaliable))

        # select n_samples data
        rnd_state = np.random.RandomState(random_state)
        self._selected = rnd_state.choice(
            list(avaliable), size=n_samples, replace=False)
        assert len(set(self._selected)) == n_samples

    @property
    def words(self) -> List[str]:
        return self._selected


# %%
def get_kmean_model(n_clusters: int):
    """
    Simple factory method to generate a kmean clusterer with pre-config params
    """
    ret = KMeans(init="k-means++", n_clusters=n_clusters,
                 random_state=10, verbose=0)
    return ret

# %%


class PlotConfig:
    """
    Plotting related configuration
    """
    figsize = (18, 7)


plt_cfg = PlotConfig()

# %%
# load the glove data
glove = GloVe("glove/glove.6B.300d.txt")
# make a sampler
sampler = WordSampler("./words_alpha.txt",
                      incl_words=glove.words, n_samples=10000)
# sample words from the sampler
sampled_words = sampler.words
# set our embedding model use sampled words
glove.words = sampled_words

# %%
# Obtain the embedding vectors from the sampled words
emb_vecs = glove.get_emb_vecs_of(sampled_words)


# %%
# build the data-matrix with shape: (n_samples, emb_dims)
X = np.array([emb_vecs[w] for w in sampled_words])
# L2-normalize all the vectors as we would like to use the metric: cosine distance
# See also: https://stats.stackexchange.com/a/146279
length = np.sqrt((X**2).sum(axis=1))[:, None]
X = X / length


# %% [markdown]
# ## Selecting the number of clusters
# Consider the **elbow method** and the **silhouette method** to have determine the number of clusters.
# Given the fact that the more clusters we have, the easier to assign the "concept" to each cluster.
# However, we would like to visualize the results and therefore we search the number of clusters from 3 to 25


# %%

# ----------------------------------------------------------------------
# Load or generate statistics for determining the number of clusters
# ----------------------------------------------------------------------
k_mean_stat = Path("results/kmean_cluster_err_stat.json")
if k_mean_stat.exists():
    with k_mean_stat.open("r") as f:
        df_data = json.load(f)
    list_k = df_data["n_clusters"]
    print(f"Loaded the {k_mean_stat}")
else:
    list_k = list(range(3, 15+1))
    # formulate the data and save it
    df_data = {
        "n_clusters": [],   # Number of clusters
        "wss": [],  # Within-Cluster-Sum of Squared Errors
        "mean_sil_coeff": [],  # Mean Silhouette Coefficient
        "time_used": []
    }
    for k in list_k:
        stime = time.time()
        km = get_kmean_model(k)
        labels = km.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        time_used = time.time() - stime
        print(f"n_clusters: {k}; "
              f"inertia: {km.inertia_:.3f}; "
              f"mean_sil_coeff: {silhouette_avg:.3f}; "
              f"time_used: {time_used:.2f}")
        df_data["n_clusters"].append(k)
        df_data["wss"].append(float(km.inertia_))
        df_data["mean_sil_coeff"].append(float(silhouette_avg))
        df_data["time_used"].append(time_used)

    # formulate the data and save it
    with k_mean_stat.open("w") as f:
        json.dump(df_data, f, indent=1)

# %%
df = pd.DataFrame.from_dict(df_data)
df = df.reset_index()

# %% [markdown]
# Elbow Method
# Generally speaking, the within cluster SSE decreases as the number of clusters \
# increase. We would like to find the plot of the point of inflection on the curve.
# Geometrically, we would like to find the point at which the curvature of the curve is
# maximum.
# A two-dimensional curve of a 1-d function:
#   $g(x,y) = f(x) - y = 0$
# The curvature is:
#   $\kappa = \frac{f''}{(1+{f'}^{2})^{3/2}}$
# See the knee definition in:
# https://raghavan.usc.edu//papers/kneedle-simplex11.pdf


# %%
# # Plot the line plot
# plt.subplots(figsize=plt_cfg.figsize)
# ax = sns.lineplot(x="n_clusters",
#                   y="wss", data=df)
# plt.xticks(np.arange(min(list_k), max(list_k)+1, 1))
# ax.set(xlabel="Number of clusters",
#        ylabel="Within-Cluster-Sum of Squared Errors")
# plt.tight_layout()
# plt.savefig("results/n_clusters_against_wss.png")
# plt.show()

# %%
kneedle = KneeLocator(df["n_clusters"], df["wss"],
                      S=1.0, curve='convex', direction='decreasing', online=False, interp_method="interp1d")
print("The number of cluster according to elbow method:", kneedle.knee)
print("The corresponding Within-Cluster-Sum of Squared Errors (WSS):", kneedle.knee_y)

# %%
# Plot knee
kneedle.plot_knee(figsize=plt_cfg.figsize)
plt.xlabel("Number of clusters")
plt.ylabel("Within-Cluster-Sum of Squared Errors")
plt.xticks(np.arange(min(list_k), max(list_k)+1, 1))
plt.tight_layout()
plt.savefig("results/knee.png")
plt.show()


# %%
# Plot normalized knee
kneedle.plot_knee_normalized(figsize=plt_cfg.figsize)
plt.tight_layout()
plt.savefig("results/knee_normalized.png")
plt.show()


# %% [markdown]
# ### The Silhouette Method
#
# The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).

# %%
plt.subplots(figsize=plt_cfg.figsize)
ax = sns.lineplot(x="n_clusters",
                  y="mean_sil_coeff", data=df)
ax.set(xlabel="Number of clusters",
       ylabel="Mean Silhouette Coefficient")
plt.xticks(np.arange(min(list_k), max(list_k)+1, 1))
plt.tight_layout()
plt.savefig("results/n_clusters_against_silhouette_score.png")
plt.show()

# %% [markdown]
# The Silhouette Method suggests number of clusters to be 7
# Combine two information, I would show the clustering with n_clusters = 7


# %%
# ----------------------
# Do clustering
# ----------------------
n_clusters = 5
km = get_kmean_model(n_clusters)
labels = km.fit_predict(X)

# %%
# build a dataframe which has word and labels
data = pd.DataFrame({
    "word": sampled_words,
    "label": labels
})
# samples words from each cluster
sampled_word_df = (data.sample(frac=1, random_state=0)   # shuffle
                       .groupby("label", sort=False).head(10))

for i in range(n_clusters):
    words = sampled_word_df[sampled_word_df["label"] == i]["word"].to_list()
    print(i, ":", ", ".join(words))

# %%
# find the cluster centers and 5 words around each center
label_to_word = dict()
for i, v in enumerate(km.cluster_centers_):
    nearests = glove.find_nearest_by_vec(v)
    print("nearest words and distances: ", nearests)
    # save down the cluster center to word
    # use it as the label when plotting out the data
    word = nearests[0][0]
    label_to_word[i] = word

# --------------------------------------------------------
# %%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)


# %%
# add pca_data for viualization
data["pca_x"] = pca_result[:, 0]
data["pca_y"] = pca_result[:, 1]


# %%
def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, label_to_txt: dict) -> Axes:
    """
    Helper function the plot the scatter plot
    """
    n_clusters = len(df["label"].unique())
    fig, ax = plt.subplots(figsize=(16, 10))

    scatter_palette = sns.color_palette("hls", n_clusters)
    txt_palette = sns.color_palette("hls", n_clusters, desat=0.6)
    for i in range(n_clusters):
        plt.scatter(
            x=df.loc[df['label'] == i, x_col],
            y=df.loc[df['label'] == i, y_col],
            color=scatter_palette[i],
            alpha=0.1)
        # find the location of the text
        xtext, ytext = df.loc[df['label'] == i, [x_col, y_col]].mean()
        # set up the box around the text
        bbox_props = dict(boxstyle="round,pad=0.3",
                          fc=txt_palette[i], alpha=0.8, lw=1)
        plt.annotate(label_to_txt[i], (xtext, ytext),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, color='k', bbox=bbox_props, alpha=0.8)
    return ax


# %%
plot_scatter(data, "pca_x", "pca_y", label_to_word)
plt.title(f'Visualize k-means clustering by PCA with {n_clusters} clusters')
plt.tight_layout()
plt.savefig("results/pca_visual.png")
plt.show()


# %%
time_start = time.time()
# More info on how to grab the perplexity and iterations:
#
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=500)
tsne_result = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# %%
data["tsne_x"] = tsne_result[:, 0]
data["tsne_y"] = tsne_result[:, 1]

plot_scatter(data, "tsne_x", "tsne_y", label_to_word)
plt.title(f'Visualize k-means clustering by T-SNE with {n_clusters} clusters')
plt.tight_layout()
plt.savefig("results/tsne_visual.png")
plt.show()
