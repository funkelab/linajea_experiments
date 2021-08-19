import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from scipy.ndimage import center_of_mass
import os
import argparse


def extract_tracks(directory, outfile):

    files = sorted(glob.glob(directory + '/*.tif'))
    print(files)

    prev_centers = {}

    next_node_id = 1

    with open(outfile, 'w') as of:

        # for each frame -> t:
        for t, f in tqdm(enumerate(files)):

            labels = imread(f)
            # print(labels.shape)
            # print(labels.dtype)
            # print(labels.min(), labels.max())

            # find centers of mass for each "label" -> z, y, x
            label_ids = list(np.unique(labels))
            label_ids.remove(0)
            num_nodes = len(label_ids)
            tqdm.write(f"label_ids={label_ids}")
            tqdm.write(f"num_nodes={num_nodes}")
            centers = {
                label_id: (node_id, (t,) + center)
                for label_id, center, node_id in zip(
                    label_ids,
                    center_of_mass(
                        np.ones(labels.shape),
                        labels,
                        label_ids),
                    range(next_node_id, next_node_id + num_nodes)
                )
            }
            next_node_id += num_nodes

            # find parent node in previous frame
            parents = {
                node_id: prev_centers.get(label_id, (-1, None))[0]
                for label_id, (node_id, _) in centers.items()
            }
            prev_centers = centers

            # write to tracks file:
            # t, z, y, x, node ID, parent ID, label
            for label_id, (node_id, center) in centers.items():
                of.write("%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (
                    int(center[0]),
                    int(center[1]*5),
                    int(center[2]),
                    int(center[3]),
                    node_id,
                    parents[node_id],
                    label_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("path_to_test")
    args = parser.parse_args()
    directory = os.path.join(args.path_to_test, '0%s_GT/TRA/' % args.dataset)
    outfile = 'seeds_test_0%s.txt' % args.dataset
    extract_tracks(directory, outfile)
