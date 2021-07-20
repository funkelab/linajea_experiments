import logging
import csv
import numpy as np
import linajea
import linajea.tracking
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)
db_host = "localhost"  # TODO: replace with mongoDB URL


def load_nodes(db_name):
    db = linajea.CandidateDatabase(db_name, db_host, 'r')
    roi = db.get_nodes_roi()
    ns = db.read_nodes(roi)
    logger.info("Found %d nodes in database %s" % (len(ns), db_name))
    nodes = {}
    for n in ns:
        t = n['t']
        nodes.setdefault(t, [])
        nodes[t].append(n)
    return nodes


def get_matches_by_radius(
        gt_kd_trees_by_frame,
        kd_trees_by_frame,
        radius):
    num_matches = 0
    for frame, gt_kd_tree in gt_kd_trees_by_frame.items():
        if frame not in kd_trees_by_frame:
            continue
        kd_tree = kd_trees_by_frame[frame]
        neighbors = gt_kd_tree.query_ball_tree(kd_tree, radius)
        for _id, x_matches in enumerate(neighbors):
            if len(x_matches) > 0:
                num_matches += 1
    logger.info("Radius %d had %f matched nodes" % (radius, num_matches))
    return num_matches


def get_candidate_recall(
        db_name, gt_db_name, radius):
    gt_nodes_by_frame = load_nodes(gt_db_name)
    nodes_by_frame = load_nodes(db_name)
    num_gt_nodes = 0
    gt_ids_to_nodes = {}
    for frame, nodes in gt_nodes_by_frame.items():
        num_gt_nodes += len(nodes)
        for node in nodes:
            gt_ids_to_nodes[node['id']] = node

    gt_kd_trees_by_frame = {}
    for frame, nodes in gt_nodes_by_frame.items():
        gt_kd_trees_by_frame[frame] = cKDTree([[c['z'], c['y'], c['x']]
                                               for c in nodes])
    trees_by_frame = {}
    for t, points in nodes_by_frame.items():
        trees_by_frame[t] = cKDTree([[c['z'], c['y'], c['x']] for c in points])
    num_matches = get_matches_by_radius(
        gt_kd_trees_by_frame,
        trees_by_frame,
        radius)
    return num_matches, num_gt_nodes


def write_candidate_recall():
    fieldnames = ['name', 'matches', 'gt_nodes', 'recall']
    with open('candidate_recall.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        dbs = {
            'mouse_early_1': 'linajea_140521_setup11_simple_late_early_400000_te',
            'mouse_early_2': 'linajea_140521_setup11_simple_middle_early_400000_te',
            'mouse_middle_1': 'linajea_140521_setup11_simple_late_middle_400000_te',
            'mouse_middle_2': 'linajea_140521_setup11_simple_early_middle_400000_te',
            'mouse_late_1': 'linajea_140521_setup11_simple_middle_late_400000_te',
            'mouse_late_2': 'linajea_140521_setup11_simple_early_late_400000_te',
            'dros_es1': 'linajea_120828_setup211_simple_eval_side_1_400000_te',
            'dros_es2': 'linajea_120828_setup211_simple_eval_side_2_400000_te',
            'zebra_es1': 'linajea_160328_setup111_simple_eval_side_1_400000_te',
            'zebra_es2': 'linajea_160328_setup111_simple_eval_side_2_400000_te',
            }
        default_radius = 15
        mouse_radius = 20
        gt_dbs = {
                'mouse_early_1': 'linajea_140521_gt_early',
                'mouse_early_2': 'linajea_140521_gt_early',
                'mouse_middle_1': 'linajea_140521_gt_middle',
                'mouse_middle_2': 'linajea_140521_gt_middle',
                'mouse_late_1': 'linajea_140521_gt_late',
                'mouse_late_2': 'linajea_140521_gt_late',
                'dros_es1': 'linajea_120828_gt_side_1',
                'dros_es2': 'linajea_120828_gt_side_2',
                'zebra_es1': 'linajea_160328_gt_side_1',
                'zebra_es2': 'linajea_160328_gt_side_2',
                }
        for name in dbs.keys():
            if 'mouse' in name:
                radius = mouse_radius
            else:
                radius = default_radius
            num_matches, num_gt_nodes = get_candidate_recall(
                    dbs[name], gt_dbs[name], radius)
            row = {
                    'name': name,
                    'matches': num_matches,
                    'gt_nodes': num_gt_nodes,
                    'recall': num_matches / num_gt_nodes
                  }
            writer.writerow(row)


def get_pv_distance(
        kd_trees_by_frame,
        gt_track_graph,
        nodes_by_frame,
        gt_nodes_by_frame,
        eval_radius):
    prediction_distances = []
    baseline_distances = []

    for frame, kd_tree in kd_trees_by_frame.items():
        if frame not in gt_nodes_by_frame:
            continue
        gt_nodes = gt_nodes_by_frame[frame]
        candidate_nodes = nodes_by_frame[frame]
        for gt_node in gt_nodes:
            gt_node_location = [gt_node['z'], gt_node['y'], gt_node['x']]
            logger.debug("Querying kd tree for gt node %s" % gt_node)
            distance, index = kd_tree.query(gt_node_location, k=1,
                                            distance_upper_bound=eval_radius)
            logger.debug("Distance: %f, Index: %d" % (distance, index))
            if distance > eval_radius:
                continue
            candidate_node = candidate_nodes[index]
            candidate_node_location = np.array([candidate_node['z'],
                                                candidate_node['y'],
                                                candidate_node['x']])
            logger.debug("CANDIDATE NODE LOC: %s" % candidate_node_location)
            parent_vector = np.array(candidate_node['parent_vector'])
            logger.debug("PARENT VECTOR: %s" % parent_vector)
            predicted_location = candidate_node_location + parent_vector
            logger.debug("PREDICTED LOCATION: %s" % predicted_location)
            gt_node_id = gt_node['id']
            parent_edges = list(gt_track_graph.prev_edges(gt_node_id))
            if len(parent_edges) > 0:
                assert len(parent_edges) == 1
                parent_id = parent_edges[0][1]
                parent_node = gt_track_graph.nodes[parent_id]
                logger.debug("PARENT_NODE: %s" % parent_node)
                parent_location = np.array([parent_node['z'],
                                            parent_node['y'],
                                            parent_node['x']])
                prediction_distance = np.linalg.norm(
                        np.array(predicted_location) - np.array(parent_location))
                logger.debug("PREDICTION DISTANCE: %s",
                             prediction_distance)
                prediction_distances.append(prediction_distance)
                baseline_distance = np.linalg.norm(
                        np.array(candidate_node_location) - np.array(parent_location))
                logger.debug("BASELINE DISTNACE: %s",
                             baseline_distance)
                baseline_distances.append(baseline_distance)
    return prediction_distances, baseline_distances


def get_movement_accuracy(
        db_name, gt_db_name, radius):
    gt_nodes_by_frame = load_nodes(gt_db_name)
    nodes_by_frame = load_nodes(db_name)
    trees_by_frame = {}
    for t, points in nodes_by_frame.items():
        trees_by_frame[t] = cKDTree([[c['z'], c['y'], c['x']] for c in points])
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host, 'r')
    roi = gt_db.get_nodes_roi()
    gt_graph = gt_db[roi]
    gt_track_graph = linajea.tracking.TrackGraph(gt_graph)
    prediction_distances, baseline_distances = get_pv_distance(
            trees_by_frame,
            gt_track_graph,
            nodes_by_frame,
            gt_nodes_by_frame,
            radius)
    pred = {}
    baseline = {}
    pred['mean'] = np.mean(prediction_distances)
    baseline['mean'] = np.mean(baseline_distances)
    pred['median'] = np.median(prediction_distances)
    baseline['median'] = np.median(baseline_distances)
    pred['stddev'] = np.std(prediction_distances)
    baseline['stddev'] = np.std(baseline_distances)
    pred['min'] = np.min(prediction_distances)
    baseline['min'] = np.min(baseline_distances)
    pred['max'] = np.max(prediction_distances)
    baseline['max'] = np.max(baseline_distances)
    pred['firstQ'] = np.percentile(prediction_distances, 25)
    baseline['firstQ'] = np.percentile(baseline_distances, 25)
    pred['thirdQ'] = np.percentile(prediction_distances, 75)
    baseline['thirdQ'] = np.percentile(baseline_distances, 75)

    return pred, baseline, prediction_distances, baseline_distances


def write_movement_accuracy():
    fieldnames = ['name', 'mean', 'median', 'stddev', 'min', 'max', 'firstQ', 'thirdQ']
    with open('movement_accuracy.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        dbs = {
            'mouse_early_1': 'linajea_140521_setup11_simple_late_early_400000_te',
            'mouse_early_2': 'linajea_140521_setup11_simple_middle_early_400000_te',
            'mouse_middle_1': 'linajea_140521_setup11_simple_late_middle_400000_te',
            'mouse_middle_2': 'linajea_140521_setup11_simple_early_middle_400000_te',
            'mouse_late_1': 'linajea_140521_setup11_simple_middle_late_400000_te',
            'mouse_late_2': 'linajea_140521_setup11_simple_early_late_400000_te',
            'dros_es1': 'linajea_120828_setup211_simple_eval_side_1_400000_te',
            'dros_es2': 'linajea_120828_setup211_simple_eval_side_2_400000_te',
            'zebra_es1': 'linajea_160328_setup111_simple_eval_side_1_400000_te',
            'zebra_es2': 'linajea_160328_setup111_simple_eval_side_2_400000_te',
            }
        mouse_radius = 20
        default_radius = 15
        gt_dbs = {
                'mouse_early_1': 'linajea_140521_gt_early',
                'mouse_early_2': 'linajea_140521_gt_early',
                'mouse_middle_1': 'linajea_140521_gt_middle',
                'mouse_middle_2': 'linajea_140521_gt_middle',
                'mouse_late_1': 'linajea_140521_gt_late',
                'mouse_late_2': 'linajea_140521_gt_late',
                'dros_es1': 'linajea_120828_gt_side_1',
                'dros_es2': 'linajea_120828_gt_side_2',
                'zebra_es1': 'linajea_160328_gt_side_1',
                'zebra_es2': 'linajea_160328_gt_side_2',
                }
        for name in dbs.keys():
            if 'mouse' in name:
                radius = mouse_radius
            else:
                radius = default_radius
            pred, baseline, pred_distances, baseline_distances = get_movement_accuracy(
                    dbs[name], gt_dbs[name], radius)
            with open('movement_distances_' + name + '.txt', 'w') as distance_file:
                for d in pred_distances:
                    distance_file.write(str(d) + '\n')
            with open('movement_distances_' + name + '_baseline.txt', 'w') as distance_file:
                for d in baseline_distances:
                    distance_file.write(str(d) + '\n')

            pred['name'] = name
            writer.writerow(pred)
            baseline['name'] = name + '_baseline'
            writer.writerow(baseline)


if __name__ == '__main__':
    write_candidate_recall()
    write_movement_accuracy()
