import argparse
import os
import linajea
import daisy
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_kd_trees_by_frame(graph):

    nodes_by_frame = {}
    node_locs_by_frame = {}
    for node_id, node in graph.nodes(data=True):
        if 't' not in node:
            logger.debug("(%d, %s) has no time, skipping" % (node_id, node))
            continue
        t = node['t']
        if t not in nodes_by_frame:
            nodes_by_frame[t] = []
            node_locs_by_frame[t] = []
        node_locs_by_frame[t].append([node['z'], node['y'], node['x']])
        nodes_by_frame[t].append(node_id)

    kd_trees_by_frame = {}
    for frame, nodes in node_locs_by_frame.items():
        kd_trees_by_frame[frame] = cKDTree(nodes)
    return nodes_by_frame, kd_trees_by_frame


def get_node_loc(graph, node_id):
    node_data = graph.nodes(data=True)[node_id]
    if 't' not in node_data:
        logger.warn("(%d, %s) has no time, skipping" % (node_id, node_data))
        return None
    return [node_data['z'], node_data['y'], node_data['x']]


def get_cell_cycle_label(graph, node_id):
    if graph.in_degree(node_id) == 2:
        return "division"
    else:
        out_edges = list(graph.out_edges(node_id))
        if len(out_edges) > 0:
            parent_id = out_edges[0][1]
            if graph.in_degree(parent_id) == 2:
                return "child"
    return "continuation"


def match_candidates(
        cand_db_name,
        gt_db_name,
        db_host,
        frames,
        match_distance,
        exclude=None,
        roi=daisy.Roi((0, 0, 0, 0), (10e10, 10e10, 10e10, 10e10))):

    if frames:
        roi = daisy.Roi((frames[0], 0, 0, 0),
                        (frames[1] - frames[0], 10e10, 10e10, 10e10))
    logger.info("Using ROI %s" % str(roi))

    label_to_points = {}
    labels = ["division", "child", "continuation"]
    for label in labels:
        label_to_points[label] = []

    cand_db = linajea.CandidateDatabase(cand_db_name, db_host)
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)

    cand_graph = cand_db[roi]
    gt_graph = gt_db[roi]

    cand_nodes, cand_kd_trees = get_kd_trees_by_frame(cand_graph)
    gt_nodes, gt_kd_trees = get_kd_trees_by_frame(gt_graph)

    logger.info("Processing frames")
    for frame, nodelist in gt_nodes.items():
        if exclude and (frame >= exclude[0] and frame < exclude[1]):
            logger.info("Excluding frame %d" % frame)
            continue
        cand_kd_tree = cand_kd_trees[frame]
        for gt_node in nodelist:
            gt_node_loc = get_node_loc(gt_graph, gt_node)
            if gt_node_loc is None:
                print("No location found for gt node, skipping")
                continue
            dist, index = cand_kd_tree.query(gt_node_loc)
            if dist < match_distance:
                # match found!
                cand_id = cand_nodes[frame][index]
                cand_loc = [frame] + get_node_loc(cand_graph, cand_id)
                if cand_loc is None:
                    print("No location found for candidate node, skipping")
                    continue
                label = get_cell_cycle_label(gt_graph, gt_node)
                label_to_points[label].append(cand_loc)
    return label_to_points


def write_results(label_to_points, outfile):
    logger.info("Writing results to %s" % outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        for label in label_to_points.keys():
            for point in label_to_points[label]:
                to_write = [label]
                to_write.extend(point)
                string = map(str, to_write)
                f.write(' '.join(string))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--cand_db',
            help="Name of database to read candidates from")
    parser.add_argument(
            '-f', '--frames', type=int,
            nargs=2, help='Start and end frames')
    parser.add_argument(
            '-e', '--exclude', type=int,
            nargs=2, help='Exclude frames start and end',
            default=None)
    parser.add_argument(
            '-g', '--gt_db', help='Gt database name')
    parser.add_argument(
            '-o', '--outfile')
    parser.add_argument(
            '-m', '--match_distance', type=int)
    parser.add_argument(
            '--db_host', default="mongodb://linajeaAdmin:FeOOHnH2O" +
            "@funke-mongodb4/admin?replicaSet=rsLinajea")
    args = parser.parse_args()
    label_to_points = match_candidates(
            args.cand_db,
            args.gt_db,
            args.db_host,
            args.frames,
            args.match_distance,
            exclude=args.exclude)
    write_results(label_to_points, args.outfile)
