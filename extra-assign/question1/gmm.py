#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.glove import GloVe
from utils.word_sample import WordSample
import numpy as np
from sklearn.mixture import GaussianMixture


# In[2]:


n_samples = 10000

glove = GloVe("glove/glove.6B.300d.txt")
words = WordSample("./words_alpha.txt", incl_words=glove.wordset, n_samples=n_samples).words

emb_vecs = glove.get_emb_vecs_of(words)
# build the data-matrix with shape = (n_samples, emb_dims)
X = np.array([emb_vecs[w] for w in words])
# normalize it
length = np.sqrt((X**2).sum(axis=1))[:, None]
X = X / length


# In[3]:


gmm = GaussianMixture(n_components=10, verbose=1)
gmm.fit(X)


# In[4]:


labels = gmm.predict(X)


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)


# In[15]:


pca.explained_variance_ratio_


# In[16]:


import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=pca_result[:, 0], y=pca_result[:, 1],
    hue=labels,
    palette=sns.color_palette("hls", 10),
    alpha=0.3
)


# In[10]:


import time
from sklearn.manifold import TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[11]:


plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3
)


#  $e^{i\pi} + 1 = 0$

# $\kappa = \frac{f''}{(1+{f'}^{2})^{3/2}}$

# In[ ]:




