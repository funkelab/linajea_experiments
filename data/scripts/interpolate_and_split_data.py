#LR_split

from TGMMlibraries import lineageTree # Personal lineage tree data structure you can find it there: https://github.com/leoguignard/TGMMlibraries
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.interpolate import InterpolatedUnivariateSpline

### TO FILL!
zebra_data = '/path/to/mamut/data-mamut_final.xml'
lT = lineageTree(zebra_data, MaMuT=True)

# setting up some variables to get better id numbers for new nodes
lT.max_id = np.max(lT.nodes)
lT.next_id = list(set(range(0, lT.max_id+1)).difference(lT.nodes))
# getting the first cells of each cell cycle/track
init_c = [c for c in lT.nodes if not c in lT.predecessor or len(lT.successor[lT.predecessor[c][0]])==2]
for c in init_c:
    track = lT.get_cycle(c) # get the track of c: a cell right after
                            # division to just before the next division, it's ordered in time
    track = [[ci] for ci in track] # ease the writing of the predecessors/successors (they are lists)
    
    # add the mother cell and the two daughter cells if existing in the track to ensure the linkage between cells
    if track[0][0] in lT.predecessor:
        track = [lT.predecessor.get(track[0][0])] + track
    if track[-1][0] in lT.successor:
        track += [lT.successor[track[-1][0]]]
    
    track = np.array(track)
    times = np.array([lT.time[ci[0]] for ci in track]) # get the times of each cell id in the track
    if ((times[:-1] - times[1:])!=-1).any(): # check the time difference between consecutive ids
        # Build piecewise linear interpolation functions independently for x, y and z coordinates
        X, Y, Z = zip(*np.array([np.mean([lT.pos[cii] for cii in ci], axis=0) for ci in track]))
        final_times = np.arange(times.min(), times.max()+1)
        X_i = InterpolatedUnivariateSpline(times, X, k=1)
        Y_i = InterpolatedUnivariateSpline(times, Y, k=1)
        Z_i = InterpolatedUnivariateSpline(times, Z, k=1)
        
        # Check for the missing times and gather them by windows of continuous missing times
        missing_times = []
        i=0
        while i < len(final_times):
            tmp = []
            while not final_times[i] in times:
                tmp += [final_times[i]]
                i+=1
            if tmp == []:
                i+=1
            else:
                missing_times += [tmp]

        for mini_track in missing_times:
            # save the first and last cell ids before missing times
            pred = track[times==mini_track[0]-1][0][0]
            succ = lT.successor[pred]

            # create a new cell for each missing time, maintaining pred and succ information
            # and interpolating the time
            for t in mini_track:
                id_ = lT.get_next_id()
                lT.pos[id_] = np.array([X_i(t), Y_i(t), Z_i(t)])
                lT.successor[pred] = [id_]
                lT.predecessor[id_] = [pred]
                lT.time[id_] = t
                lT.time_nodes[t] += [id_]
                pred = id_
                lT.nodes += [id_]
            lT.successor[id_] = succ
            for si in succ:
                lT.predecessor[si] = [id_]

# To split our dataset into two, we first reorient it with a PCA
# On the 3D position of the last nodes that are align along the midline of the embryo
pca = PCA(3)

last_nodes = list(set(lT.predecessor).difference(lT.successor))
last_pos = [lT.pos[c] for c in last_nodes]
pca.fit(last_pos)
pos = [lT.pos[c] for c in lT.nodes]
new_pos = pca.transform(pos)
lT.new_pos = dict(zip(lT.nodes, new_pos))

# These are just for visualisation purposes
first_nodes = set(lT.successor).difference(lT.predecessor)
first_nodes = np.array(list(first_nodes))

first_pos = [lT.new_pos[c] for c in first_nodes]

first_pos = np.array(first_pos)

L = []#first_pos[first_pos[:,1] < 0]
R = []#first_pos[first_pos[:,1] >= 0]

L_nodes = []
R_nodes = []
D_nodes = []
succ = lT.successor
for c in first_nodes:
    track = [c]
    while track[-1] in succ:
        track += [succ[track[-1]][0]]
    if lT.new_pos[track[0]][1]<0 and lT.new_pos[track[-1]][1]<0:
        L_nodes += [c]
        L += [lT.new_pos[c]]
    elif 0<=lT.new_pos[track[-1]][1] and 0<=lT.new_pos[track[0]][1]:
        R_nodes += [c]
        R += [lT.new_pos[c]]
    else:
        D_nodes += [c]
    t_id += 1


# 3D plot of the first cell positions,
# colored according to there positions in the PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*zip(*R))
ax.scatter(*zip(*L))
min_ = np.min(lT.new_pos.values())
max_ = np.max(lT.new_pos.values())
ax.set_xlim(min_, max_)
ax.set_ylim(min_, max_)
plt.show()

# 2D projection of all the tracks:
# Black: Discarded
# Magenta: Left
# Cyan: Right
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
for n in first_nodes:
    track = [n]
    if n in L_nodes:
        color = 'm'
    elif n in R_nodes:
        color = 'c'
    else:
        color = 'k'
    while track[-1] in lT.successor:
        track += [lT.successor[track[-1]][0]]
    pos_track = [lT.new_pos[c][:2] for c in track]
    ax.plot(*zip(*pos_track), marker = '', color = color)
    ax.plot(*zip(pos_track[0]), marker = 'o', color = color)
    ax.plot(*zip(pos_track[-1]), marker = '>', color = color)
min_ = np.min(lT.new_pos.values())
max_ = np.max(lT.new_pos.values())
ax.set_xlim(min_, max_)
ax.set_ylim(min_, max_)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# Write the three output files
init_cells = first_nodes
succ = lT.successor
line = '{:d}\t{:f}\t{:f}\t{:f}\t{:d}\t{:d}\t{:d}\n' # format of the output line
L = open('train_set.txt', 'w')
R = open('test_set.txt', 'w')
Disc = open('Discarded.txt', 'w')
t_id = 0
tot_print = 0
for c in init_cells:
    track = []
    to_do = [c]
    while to_do != []:
        track += [to_do.pop()]
        to_do += succ.get(track[-1], [])
    tot_print += len(track)
    if lT.new_pos[track[0]][1]<0 and lT.new_pos[track[-1]][1]<0:
        for ci in track:
            L.write(line.format(*(lT.time[ci],) + tuple(lT.pos[ci][::-1]) + (ci, lT.predecessor.get(ci, [-1])[0], t_id)))
    elif 0<=lT.new_pos[track[-1]][1] and 0<=lT.new_pos[track[0]][1]:
        for ci in track:
            R.write(line.format(*(lT.time[ci],) + tuple(lT.pos[ci][::-1]) + (ci, lT.predecessor.get(ci, [-1])[0], t_id)))
    else:
        for ci in track:
            Disc.write(line.format(*(lT.time[ci],) + tuple(lT.pos[ci][::-1]) + (ci, lT.predecessor.get(ci, [-1])[0], t_id)))
    t_id += 1

L.close()
R.close()
Disc.close()