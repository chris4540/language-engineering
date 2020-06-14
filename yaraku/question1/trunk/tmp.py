import json
import matplotlib.pyplot as plt


def get_1st_deriviatives(x_to_y: dict):
    ret = dict()
    x_arr = sorted([int(x) for x in x_to_y.keys()])

    # back-ward difference
    for i in range(1, len(x_arr)-1):
        b = x_arr[i+1]
        a = x_arr[i-1]
        # ---------------
        d = x_to_y[b] - x_to_y[a]
        h = b - a
        ret[x_arr[i]] = d / h
    return ret


with open("sse.json", "r") as f:
    sse_ = json.load(f)

n_clusters = sorted([int(k) for k in sse_.keys()])
sse = {int(k): v for k, v in sse_.items()}
slope = get_1st_deriviatives(sse)
print(slope)

plt.style.use('fivethirtyeight')
plt.figure()
fig, ax = plt.subplots()
fig.set_size_inches(18, 7)
x_arr = sorted(slope.keys())
y_arr = [slope[x] for x in x_arr]
ax.plot(x_arr, y_arr, '-x')
ax.set_xlabel(r'Number of clusters *k*')
ax.set_ylabel('1st deriv. of sum of squared distance')
# ---------------------------------------
plt.tight_layout()
plt.savefig("elbow_1st_d.png")
