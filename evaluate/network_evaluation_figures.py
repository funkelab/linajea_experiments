# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import csv
import matplotlib.pyplot as plt


def load_candidate_recall(file):
    x = []
    y = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(row['name'])
            y.append(float(row['recall']))
    return (x, y)


x, y = load_candidate_recall('candidate_recall.csv')
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)
plt.bar(x, y)
plt.ylim([0.95, 1])
plt.title("Ground Truth Cell Recall")
plt.savefig("candidate_recall.png")
plt.show()


def load_movement_accuracy(file):
    metric = 'mean'
    x = []
    y = []
    baseline = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'baseline' in row['name']:
                baseline.append(float(row[metric]))
            else:
                x.append(row['name'])
                y.append(float(row[metric]))
    return(x, y, baseline)


# +
x, y, baseline = load_movement_accuracy('movement_accuracy.csv')

plt.figure().set_figwidth(10)
plt.scatter(x, y, label='ours')
plt.scatter(x, baseline, label='baseline')
plt.legend()
plt.show()


# -

def load_distances(file):
    distances = []
    with open(file, 'r') as f:
        for row in f.readlines():
            distances.append(float(row))
    return distances


# +
from matplotlib.patches import Patch
filename = 'movement_distances_%s.txt'
baseline_filename = 'movement_distances_%s_baseline.txt'
labels = [
    'mouse_early',
    'mouse_middle',
    'mouse_late',
    'dros_es1',
    'dros_es2',
    'zebra_es1',
    'zebra_es2',
         ]
distances = []
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)
ax = plt.axes()
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels)
ax.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel("Distance (pixels)")
plt.title("Distance Between Predicted and Actual Parent Location")

baseline_distances = []
for label in labels:
    baseline_distances.append(load_distances(baseline_filename % label))

parts = plt.violinplot(baseline_distances, showmedians=True)
baseline_color = parts['bodies'][0].get_facecolor()

for label in labels:
    distances.append(load_distances(filename % label))

parts = plt.violinplot(distances, showmedians=True)

for pc in parts['bodies']:
    pc.set_facecolor('orange')
plt.legend([Patch(color='orange'), Patch(color=baseline_color)], ['ours', 'baseline'])
plt.savefig('movement_distance_placeholder.png')
