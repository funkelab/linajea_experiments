from linajea import load_config, tracking_params_from_config
from linajea.evaluation.match_nodes import match
import argparse
import linajea
import logging
import numpy as np
import scipy.spatial

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def load_seeds(filename):
    seeds = []  # z y x
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            point = tokens[1:4]
            seeds.append(list(map(int, point)))
    return seeds


def select_from_seeds(graph, seeds, matching_threshold=20):

    # prepare KD trees

    graph_kd_trees = {}
    begin, end = graph.get_frames()

    positions = {}
    node_ids = {}
    for t in range(begin, end):

        logger.info("Creating KD tree for frame %d...", t)

        frame_nodes = graph.cells_by_frame(t)
        positions[t] = [
            [graph.nodes[n]['z'],
             graph.nodes[n]['y'],
             graph.nodes[n]['x']]
            for n in frame_nodes
        ]
        node_ids[t] = frame_nodes

        graph_kd_trees[t] = scipy.spatial.cKDTree(positions[t])

    # find starting points in graph closest to seeds

    match_costs = {}
    for i, seed in enumerate(seeds):

        js = graph_kd_trees[begin].query_ball_point(
            seed,
            matching_threshold)

        for j in js:
            match_costs[(i, j)] = np.linalg.norm(
                np.array(seed) -
                np.array(positions[begin][j]))

    no_match_cost = max([c for c in match_costs.values()]) * 2
    matches, cost = match(match_costs, no_match_cost)

    for i, j in matches:
        logger.debug(
            "Seed %d matches with node %d, distance = %.2f",
            i, node_ids[begin][j],
            match_costs[(i, j)])

    seed_node_ids = list([node_ids[begin][j] for _, j in matches])

    # get initial tracks

    def traverse_track(node_id):

        next_edges = list(graph.next_edges(node_id))
        logger.debug(f"Node {node_id}, next edges: {next_edges}")
        if not next_edges:
            return [], [node_id]

        track = []
        leave_nodes = []

        for edge in next_edges:
            sub_track, sub_leave_nodes = traverse_track(edge[0])
            track.append(edge)
            track += sub_track
            leave_nodes += sub_leave_nodes

        return track, leave_nodes

    tracks = []
    for node_id in seed_node_ids:

        for u, v in graph.edges:
            if u == node_id or v == node_id:
                logger.debug(f"({u}, {v})")

        track, leave_nodes = traverse_track(node_id)
        tracks.append(track)

        logger.debug("Found track:\n%s\nLeave nodes: %s", track, leave_nodes)

    return tracks


def track_from_seeds(config_file, seed_file):
    config = load_config(config_file)
    arguments = config['general']
    arguments.update(config['evaluate'])

    # read reconstruction
    logger.info("Opening REC database...")
    rec_db = linajea.CandidateDatabase(
        arguments['db_name'],
        arguments['db_host'],
        mode='r+')
    logger.info("Finding parameters ID...")
    tracking_params = tracking_params_from_config(config)
    parameters_id = rec_db.get_parameters_id(tracking_params)
    rec_db.set_parameters_id(parameters_id)
    total_roi = rec_db.get_nodes_roi()
    logger.info("Reading REC graph in %s in db %s with parameters id %d ...",
                total_roi, arguments['db_name'], parameters_id)
    rec_graph = rec_db.get_selected_graph(total_roi)
    logger.info("Read %d nodes and %d edges",
                rec_graph.number_of_nodes(), rec_graph.number_of_edges())
    rec_track_graph = linajea.tracking.TrackGraph(rec_graph)

    # get seeds
    if seed_file:
        seeds = load_seeds(seed_file)
    else:
        # read GT
        logger.info("Opening GT database...")
        gt_db = linajea.CandidateDatabase(
            arguments['gt_db_name'],
            arguments['db_host'])
        logger.info("Reading GT graph...")
        gt_graph = gt_db[total_roi]
        gt_track_graph = linajea.tracking.TrackGraph(gt_graph)
        seed_ids = gt_track_graph.cells_by_frame(
                gt_track_graph.get_frames()[0])

        seeds = list([
            tuple(gt_track_graph.nodes[i][t] for t in ['z', 'y', 'x'])
            for i in seed_ids
        ])

    logger.debug(seeds)

    logger.info("Finding tracks from seeds...")
    tracks = select_from_seeds(rec_track_graph, seeds)

    selected_attr = f'selected_from_seeds_{parameters_id}'
    for (u, v) in rec_graph.edges:
        rec_graph.edges[u, v][selected_attr] = False
    for track in tracks:
        for (u, v) in track:
            rec_graph.edges[u, v][selected_attr] = True

    logger.info("Storing tracks as %s...", selected_attr)
    rec_graph.update_edge_attrs(total_roi, attributes=[selected_attr])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="config file")
    parser.add_argument('-s', '--seed-file', default=None,
                        help="tracks file with seed points")
    args = parser.parse_args()
    config_file = args.config
    seed_file = args.seed_file
    track_from_seeds(config_file, seed_file)
