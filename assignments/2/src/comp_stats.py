import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse labels (dataset names).
with open('seq.txt', 'r') as f:
    labels = list(map(lambda x: x.strip(), f.readlines()))

# Parse numbers of failures for each tracker.
n_failures1 = np.loadtxt("./n_failures_ms.txt").astype(int)
n_failures2 = np.loadtxt("./n_failures_ncc.txt").astype(int)

# Plot results as a bar plot.
fig, ax = plt.subplots()
width=0.35
rects1 = ax.bar(np.arange(len(labels)) - width/2, n_failures1, width, label='mean-shift tracker')
rects2 = ax.bar(np.arange(len(labels)) + width/2, n_failures2, width, label='normalized cross-correlation tracker')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=45)
plt.ylabel('number of failures')
plt.show()

