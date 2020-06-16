# Use vscode interactive and then convert this file to jupyter
# See also: https://code.visualstudio.com/docs/python/jupyter-support-py#_export-a-jupyter-notebook

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from utils.word_sample import WordSample
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict
from sklearn.neighbors import NearestNeighbors
get_ipython().run_line_magic('matplotlib', 'inline')


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
    def wordset(self) -> List[str]:
        ret = list(self._words)
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


glove = GloVe("glove/glove.6B.300d.txt")
# %%
words = WordSample("./words_alpha.txt", incl_words=glove.wordset, n_samples=10000).words


# %%
emb_vecs = glove.get_emb_vecs_of(words)
# build the data-matrix with shape = (n_samples, emb_dims)
X = np.array([emb_vecs[w] for w in words])
# normalize it
length = np.sqrt((X**2).sum(axis=1))[:, None]
X = X / length
# TODO: add more notes why we use normalized (cosine distacne)


# %%
def get_kmean_model(n_clusters):
    """
    Factory function to give the kmean clusterer out
    """
    ret = KMeans(init="k-means++", n_clusters=n_clusters, random_state=10, verbose=0)
    return ret

# %% [markdown]
# ## Selecting the number of clusters
# Consider the elbow method and the silhouette method to have determine the number of clusters.
# Given the fact that the more clusters we have, the easier to assign the "concept" to each cluster.
# However, we would like to visualize the results and therefore we search the number of clusters from 2 to 20

# %%
from sklearn.metrics import silhouette_samples, silhouette_score
sse = dict()
silhouette_coffs = dict()
list_k = list(range(2, 20+1))
for k in list_k:
    stime = time.time()
    clusterer = get_kmean_model(k)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    time_used = time.time() - stime
    print(f"n_clusters: {k}; inertia: {clusterer.inertia_}; silhouette_avg: {silhouette_avg}; time_used: {time_used}")
    sse[k] = clusterer.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    silhouette_coffs[k] = silhouette_avg


# %%
# make a dataframe for seaborn plotting
df_data = {
    "Number of clusters": list_k,
    "Within-Cluster-Sum of Squared Errors": [sse[k] for k in list_k],
    "Mean Silhouette Coefficient": [silhouette_coffs[k] for k in list_k]
}
df = pd.DataFrame.from_dict(df_data)
df = df.reset_index()

# %% [markdown]
# ### Elbow Method

# %%
sns.set_style('darkgrid')
sns.set_palette('muted')


# %%
figsize = (18, 7)
plt.subplots(figsize=figsize)
ax = sns.lineplot(x="Number of clusters", y="Within-Cluster-Sum of Squared Errors", data=df)
x = list_k
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.show()


# %%
# use KneeLocator to find the number of clusters
# the definition is mentioned in https://raghavan.usc.edu//papers/kneedle-simplex11.pdf page 2
kneedle = KneeLocator(df["Number of clusters"], df["Within-Cluster-Sum of Squared Errors"], S=1.0, curve='convex', direction='decreasing', online=False, interp_method="interp1d")


# %%
kneedle.plot_knee(figsize=figsize)
plt.xlabel("Number of clusters")
plt.ylabel("Within-Cluster-Sum of Squared Errors")
x = list_k
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.show()


# %%
kneedle.plot_knee_normalized(figsize=figsize)
plt.show()


# %%
print(f"The number of cluster according to elbow method: {kneedle.knee}")
print(f"The corresponding Within-Cluster-Sum of Squared Errors (WSS): {kneedle.knee_y}")

# %% [markdown]
# ### The Silhouette Method
#
# The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).

# %%
figsize = (18, 7)
plt.subplots(figsize=figsize)
ax = sns.lineplot(x="Number of clusters", y="Mean Silhouette Coefficient", data=df)
x = list_k
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.show()

# %% [markdown]
# The Silhouette Method suggests number of clusters to be 2

# %%
n_clusters = 2
km = get_kmean_model(n_clusters)
labels = km.fit_predict(X)


# %%
label_to_word = dict()
for i, v in enumerate(km.cluster_centers_):
    nearests = glove.find_nearest_by_vec(v)
    print("nearest word and distance: ", nearests)
    word = nearests[0][0]
    label_to_word[i] = word


# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)


# %%
# make pca_data for viualization
data = pd.DataFrame({"labels": labels})
data["x"] = pca_result[:, 0]
data["y"] = pca_result[:, 1]
data


# %%
def plot_scatter(df, label_to_txt):
    """
    Helper function the plot the scatter plot
    """
    n_clusters = df["labels"].unique()
    fig, ax = plt.subplots(figsize=(16,10))

    scatter_palette = sns.color_palette("hls", n_clusters)
    txt_palette = sns.color_palette("hls", n_clusters, desat=0.6)
    for i in range(n_clusters):
        plt.scatter(
            x=df.loc[df['labels']==i, 'x'],
            y=df.loc[df['labels']==i, 'y'],
            color=scatter_palette[i],
            alpha=0.1)
        # find the location of the text
        xtext, ytext = df.loc[df['labels']==i, ['x', 'y']].mean()
        # set up the box around the text
        bbox_props = dict(boxstyle="round,pad=0.3", fc=txt_palette[i], alpha=0.8, lw=1)
        plt.annotate(label_to_txt[i], (xtext, ytext),
            horizontalalignment='center',
            verticalalignment='center',
            size=15, color='k', bbox=bbox_props, alpha=0.8)
    return ax


# %%
data["labels"].unique()


# %%
fig, ax = plt.subplots(figsize=(16,10))
scatter_palette = sns.color_palette("hls", n_clusters)
txt_palette = sns.color_palette("hls", n_clusters, desat=0.6)
for i in range(n_clusters):
    plt.scatter(
        x=data.loc[data['labels']==i, 'x'],
        y=data.loc[data['labels']==i, 'y'],
        color=scatter_palette[i],
        alpha=0.1)
    xtext, ytext = data.loc[data['labels']==i, ['x','y']].mean()
    bbox_props = dict(boxstyle="round,pad=0.3", fc=txt_palette[i], alpha=0.8, lw=1)
    plt.annotate(label_to_word[i], (xtext, ytext),
        horizontalalignment='center',
        verticalalignment='center',
        size=15, color='k', bbox=bbox_props, alpha=0.8)
plt.show()


# %%
import time
from sklearn.manifold import TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=400, n_iter=300)
tsne_results = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# %%
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hls", n_clusters),
    legend="full",
    alpha=0.3
)


# %%



# %%
df = pd.DataFrame.from_dict(silhouette_scores, orient="index", columns=["Mean Silhouette Coefficient"])
df.index.name = "Number of clusters"
df = df.reset_index()

