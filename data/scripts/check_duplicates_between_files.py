import linajea
from linajea.tracking import TrackGraph
import daisy
import logging
import json
import argparse
import networkx as nx
from check_ground_truth import read_nodes_and_edges

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_graph_from_db(
        gt_db_name,
        db_host,
        start_frame,
        end_frame):

    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 100000, 100000, 100000))
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)
    gt_graph = gt_db[roi]
    if not gt_graph.number_of_edges():
        logger.info("No edges in database. Skipping track formation.")
        return
    return gt_graph


def load_graph_from_file(filename):
    nodes, edges = read_nodes_and_edges(filename)
    for node, data in nodes.items():
        data['t'] = data['position'][0]
        data['z'] = data['position'][1]
        data['y'] = data['position'][2]
        data['x'] = data['position'][3]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.items())
    graph.add_edges_from([(edge['source'], edge['target'])
                          for edge in edges])
    return graph


def check_for_duplicate_gt_tracks(
        graphs,
        cell_radius,
        node_overlap_threshold):

    track_graphs = [TrackGraph(graph) for graph in graphs]
    tracks = [track_graph.get_tracks() for track_graph in track_graphs]

    logger.info("Found {} tracks".format([len(t) for t in tracks]))
    for track_set, track_graph in zip(tracks, track_graphs):
        for track_id, track in enumerate(track_set):
            for cell_id in track.nodes():
                track_graph.nodes[cell_id]['track_id'] = track_id

    # Count how many times each pair of tracks has cells close to each other
    dup_counts = {}
    tg1 = track_graphs[0]
    tg2 = track_graphs[1]
    for frame in tg1._cells_by_frame.keys():
        cells1 = tg1._cells_by_frame[frame]
        cells2 = tg2._cells_by_frame[frame]
        for i in range(len(cells1)):
            c1 = cells1[i]
            for j in range(len(cells2)):
                c2 = cells2[j]
                if 'track_id' not in tg1.nodes[c1]:
                    print(c1)
                    print(tg1.nodes[c1])
                track1 = tg1.nodes[c1]['track_id']
                track2 = tg2.nodes[c2]['track_id']
                if close(tg1.nodes[c1],
                         tg2.nodes[c2],
                         cell_radius):
                    logger.info("These cells are close together: "
                                "%d %s    %d %s"
                                % (c1, tg1.nodes[c1],
                                   c2, tg2.nodes[c2]))
                    tup = (min(track1, track2), max(track1, track2))
                    dup_counts[tup] = 1 if tup not in dup_counts\
                        else dup_counts[tup] + 1
    print(dup_counts)

    for pair in dup_counts:
        if dup_counts[pair] < node_overlap_threshold:
            continue
        print("Tracks {} and {} are duplicates".format(*pair))
        t1 = tracks[pair[0]]
        t2 = tracks[pair[1]]
        print("%d has %d cells, %d has %d cells"
              % (pair[0], t1.number_of_nodes(),
                 pair[1], t2.number_of_nodes()))
        to_delete = pair[0] if t1.number_of_nodes() < t2.number_of_nodes()\
            else pair[1]
        print("Would delete track {}".format(to_delete))


def close(c1, c2, cell_radius):
    p1 = [c1['z'], c1['y'], c1['x']]
    p2 = [c2['z'], c2['y'], c2['x']]
    assert len(p1) == len(p2)
    for dim in range(len(p1)):
        if abs(p1[dim] - p2[dim]) > cell_radius:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+')
    parser.add_argument('-r', '--radius', type=int)
    parser.add_argument('-o', '--overlap_threshold', type=int)
    parser.add_argument('-m', '--move_threshold', type=int)
    args = parser.parse_args()
    
    graphs = [load_graph_from_file(f) for f in args.filenames]
    check_for_duplicate_gt_tracks(
            graphs, args.radius, args.overlap_threshold)
