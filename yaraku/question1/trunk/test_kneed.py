import json
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import interp1d

with open("sse_minibatch.json", "r") as f:
    sse_ = json.load(f)

n_clusters = sorted([int(k) for k in sse_.keys()])
sse = {int(k): v for k, v in sse_.items()}
y = [sse[k] for k in n_clusters]
x = n_clusters
# print(x)
# f = interp1d(x, y)
# x_new = np.arange(10, max(n_clusters)+1, 5)
# print(x_new)
# y_new = f(x_new)
# plt.plot(x, y, 'o', x_new, y_new, '-')
# plt.savefig("interp1d.png")
# slope = get_1st_deriviatives(sse)
# for i, j in zip(x_new, y_new):
#     print(i,j)

# # # plt.style.use('fivethirtyeight')
kneedle = KneeLocator(x, y, S=1.0, curve='convex', direction='decreasing', online=True, interp_method="polynomial")
print(kneedle.knee)
print(kneedle.knee_y)
plt.style.use('fivethirtyeight')
kneedle.plot_knee(figsize=(18, 7))
plt.savefig("knee.png")

kneedle.plot_knee_normalized(figsize=(18, 7))
plt.savefig("knee_normal.png")
