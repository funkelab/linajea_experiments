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


def load_scores(file):
    t = []
    fractions = []
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(int(row['time']))
            fractions.append(float(row['percent']))            
    return (t, fractions)


import numpy as np
colors = ['red', 'green', 'blue']
def plot_scores(scores, title, file=None, xmax=50):
    plt.rc('font', size=15) 
    i = 0
    for name, result in scores.items():
        plt.scatter(*result, label=name, color=colors[i], s=5)
        i += 1
    plt.legend()
    plt.ylim([0,1])
    plt.xlim([0,xmax])
    plt.gca().set_yticks(np.arange(0, 1, 0.1), minor=True)
    plt.gca().set_xticks(np.arange(0, xmax, 10), minor=True)
    if xmax > 50:
        plt.gca().set_xticks(np.arange(0, xmax +1, 50), minor=False)
    
    #plt.title(title)
    plt.grid(True, linewidth=0.3, color='grey', linestyle='-')
    plt.grid(True, linewidth=0.3, color='grey', linestyle='dotted',which="minor")
    plt.xlabel("Time frames")
    plt.ylabel("Track accuracy")
    plt.tight_layout()
    if file:
        plt.savefig(file)
    plt.show()
        


dros_es1 = {
   "ours": 'drosophila_segments_eval_side_1_ours.csv',
   "greedy": 'drosophila_segments_eval_side_1_greedy.csv',
   "TGMM": 'drosophila_segments_eval_side_1_tgmm.csv',
}
dros_es1_res = {}
for name, file in dros_es1.items():
    dros_es1_res[name] = load_scores(file)
plot_scores(dros_es1_res, "Drosophila Eval Side 1", file="segments_drosophila_es1.png", xmax=150)

dros_es2 = {
   "ours": 'drosophila_segments_eval_side_2_ours.csv',
   "greedy": 'drosophila_segments_eval_side_2_greedy.csv',
   "TGMM": 'drosophila_segments_eval_side_2_tgmm.csv',
}
dros_es2_res = {}
for name, file in dros_es2.items():
    dros_es2_res[name] = load_scores(file)
plot_scores(dros_es2_res, "Drosophila Eval Side 2", file="segments_drosophila_es2.png", xmax=150)

dros_av = {
   "ours": 'dros_segments_averaged_ours.csv',
   "greedy": 'dros_segments_averaged_greedy.csv',
   "TGMM": 'dros_segments_averaged_tgmm.csv',
}
dros_av_res = {}
for name, file in dros_av.items():
    dros_av_res[name] = load_scores(file)
plot_scores(dros_av_res, "Drosophila Averaged", file="segments_drosophila_av.png", xmax=150)

zebra_es1 = {
   "ours": 'zebrafish_segments_eval_side_1_ours.csv',
   "greedy": 'zebrafish_segments_eval_side_1_greedy.csv',
   "TGMM": 'zebrafish_segments_eval_side_1_tgmm.csv',
}
zebra_es1_res = {}
for name, file in zebra_es1.items():
    zebra_es1_res[name] = load_scores(file)
plot_scores(zebra_es1_res, "Zebrafish Eval Side 1", file="segments_zebrafish_es1.png",xmax=150)

zebra_es2 = {
   "ours": 'zebrafish_segments_eval_side_2_ours.csv',
   "greedy": 'zebrafish_segments_eval_side_2_greedy.csv',
   "TGMM": 'zebrafish_segments_eval_side_2_tgmm.csv',
}
zebra_es2_res = {}
for name, file in zebra_es2.items():
    zebra_es2_res[name] = load_scores(file)
plot_scores(zebra_es2_res, "Zebrafish Eval Side 2", file="segments_zebrafish_es2.png", xmax=150)

zebra_av = {
   "ours": 'zebra_segments_averaged_ours.csv',
   "greedy": 'zebra_segments_averaged_greedy.csv',
   "TGMM": 'zebra_segments_averaged_tgmm.csv',
}
zebra_av_res = {}
for name, file in zebra_av.items():
    zebra_av_res[name] = load_scores(file)
plot_scores(zebra_av_res, "Zebrafish Averaged", file="segments_zebrafish_av.png", xmax=150)

# +
mouse_early_1 = {
   "ours": load_scores('mouse_segments_late_early_ours.csv'),
   "greedy": load_scores('mouse_segments_late_early_greedy.csv'),
   "TGMM": load_scores('mouse_segments_late_early_tgmm.csv'),
}
mouse_early_2 = {
   "ours": load_scores('mouse_segments_middle_early_ours.csv'),
   "greedy": load_scores('mouse_segments_middle_early_greedy.csv'),
   "TGMM": load_scores('mouse_segments_middle_early_tgmm.csv'),
}
  

plot_scores(mouse_early_1, "Mouse Early 1", file="segments_mouse_early_1.png")
plot_scores(mouse_early_2, "Mouse Early 2", file="segments_mouse_early_2.png")
# -

mouse_middle_1 = {
   "ours": load_scores('mouse_segments_late_middle_ours.csv'),
   "greedy": load_scores('mouse_segments_late_middle_greedy.csv'),
   "TGMM": load_scores('mouse_segments_late_middle_tgmm.csv'),
}
mouse_middle_2 = {
   "ours": load_scores('mouse_segments_early_middle_ours.csv'),
   "greedy": load_scores('mouse_segments_early_middle_greedy.csv'),
   "TGMM": load_scores('mouse_segments_early_middle_tgmm.csv'),
}
plot_scores(mouse_middle_1, "Mouse Middle 1", file="segments_mouse_middle_1.png")
plot_scores(mouse_middle_2, "Mouse Middle 2", file="segments_mouse_middle_2.png")

mouse_av = {
   "ours": 'mouse_segments_averaged_ours.csv',
   "greedy": 'mouse_segments_averaged_greedy.csv',
   "TGMM": 'mouse_segments_averaged_tgmm.csv',
}
mouse_av_res = {}
for name, file in mouse_av.items():
    mouse_av_res[name] = load_scores(file)
plot_scores(mouse_av_res, "Mouse Averaged", file="segments_mouse_av.png")

mouse_late_1 = {
   "ours": load_scores('mouse_segments_middle_late_ours.csv'),
   "greedy": load_scores('mouse_segments_middle_late_greedy.csv'),
   "TGMM": load_scores('mouse_segments_middle_late_tgmm.csv'),
}
mouse_late_2 = {
   "ours": load_scores('mouse_segments_early_late_ours.csv'),
   "greedy": load_scores('mouse_segments_early_late_greedy.csv'),
   "TGMM": load_scores('mouse_segments_early_late_tgmm.csv'),
}
plot_scores(mouse_late_1, "Mouse Late 1", file="segments_mouse_late_1.png")
plot_scores(mouse_late_2, "Mouse Late 2", file="segments_mouse_late_2.png")






