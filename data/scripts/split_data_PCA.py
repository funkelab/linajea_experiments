from sklearn.decomposition import PCA
import networkx as nx
import logging
import csv
import os
import argparse

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def load_from_file(filename):
    graph = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            try:
                tokens = list(map(int, tokens))
            except:
                # skip header and try again
                continue
            t, z, y, x, cell_id, parent_id, track_id = tokens
            graph.add_node(cell_id, t=t, z=z, y=y, x=x, parent_id=parent_id,
                    track_id=track_id, cell_id=cell_id)
            if parent_id != -1:
                graph.add_edge(cell_id, parent_id)
    return graph


def get_location(node, graph):
    data = graph.nodes[node]
    pos_attrs = ['z', 'y', 'x']
    return [data[p] for p in pos_attrs]


def split_tracks(graph, output_dir):
    # fit PCA on all nodes
    pca = PCA(3)
    positions = [get_location(node, graph) for node in graph.nodes()]
    pca.fit(positions)

    # transform all node locations into PCA space
    for node in graph.nodes:
        old_loc = get_location(node, graph)
        new_loc = pca.transform([old_loc])[0]
        graph.nodes[node]['pca_loc'] = new_loc
        graph.nodes[node]['side'] = int(new_loc[0] < 0)

    # split tracks into side1, side2, and crossing
    side1 = []
    side2 = []
    crossing = []
    tracks = [graph.subgraph(g).copy() for g in
              nx.weakly_connected_components(graph)]
    for track in tracks:
        side = None
        for node, data in track.nodes(data=True):
            if side is None:
                side = data['side']
            elif data['side'] != side:
                side = -1
                crossing.append(track)
                break

        if side == 0:
            side1.append(track)
        elif side == 1:
            side2.append(track)

    logger.info("%d tracks on side 1, %d tracks on side 2, %d crossing",
                len(side1), len(side2), len(crossing))

    # write to files
    write_tracks_to_file(side1, os.path.join(output_dir, 'tracks_side_1.csv'))
    write_tracks_to_file(side2, os.path.join(output_dir, 'tracks_side_2.csv'))
    write_tracks_to_file(crossing, os.path.join(output_dir,
        'crossing_tracks.csv'))


def write_tracks_to_file(tracks, filename):
    with open(filename, 'w') as f:
        header = ['t', 'z', 'y', 'x', 'cell_id', 'parent_id', 'track_id']
        csv_writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        csv_writer.writeheader()
        for track_id, track in enumerate(tracks):
            for node, data in track.nodes(data=True):
                # data['cell_id'] = node
                # data['parent_id'] = list(track.out_edges(node))[0][1]
                if 'track_id' not in data:
                    data['track_id'] = track_id
                csv_writer.writerow(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    graph = load_from_file(args.filename)
    split_tracks(graph, args.output_dir)
