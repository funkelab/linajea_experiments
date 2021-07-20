from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

source_name = 'source'
target_name = 'target'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def read_nodes_and_edges(filename):
    logging.info("Reading nodes and edges from %s" % filename)
    nodes = {}
    edges = []
    max_id = -1
    for line in open(filename):
        # 0  1  2  3  4        5          6
        # t, z, y, x, cell_id, parent_id, track_id
        t, z, y, x, cell_id, parent_id, track_id = line.split()
        t = int(float(t))
        z = float(z)
        y = float(y)
        x = float(x)
        cell_id = int(cell_id)
        parent_id = int(parent_id)
        track_id = int(track_id)
        if parent_id > 0:
            edges.append(
                {
                    source_name: cell_id,
                    target_name: parent_id,
                })

        nodes[cell_id] = {
                'position': (t, z, y, x),
                'id': cell_id,
            }
        max_id = max(max_id, cell_id)
    logging.info("%d nodes read" % len(nodes))
    logging.info("%d edges read" % len(edges))
    logging.info("Max cell id is %d" % max_id)

    return nodes, edges


def get_edge_distance_stats(nodes, edges, histogram=False):
    logger.info("Getting distance statistics for edges")
    distances = []
    for edge in edges:
        source_id = edge[source_name]
        target_id = edge[target_name]
        if target_id == -1 or target_id not in nodes:
            logger.warn("Target %d for edge %s not in nodes"
                        % (target_id, edge))
            continue
        source_node = nodes[source_id]
        target_node = nodes[target_id]

        distance = calc_distance(source_node['position'][1:],
                                 target_node['position'][1:])
        edge['distance'] = distance
        if distance > 40:
            logging.debug("Distance greater than 40:")
            logging.debug("Edge", edge)
            logging.debug("Source", source_node)
            logging.debug("Target", target_node)
            logging.debug("Distance", distance)

        distances.append(distance)
    if len(distances) == 0:
        logger.warn("No edges found, cannot get max distance")
        return

    max_distance = max(distances)
    max_index = distances.index(max_distance)
    longest_edge = edges[max_index]
    print("Max distance:", max_distance,
          'source', nodes[longest_edge[source_name]],
          'target', nodes[longest_edge[target_name]])
    print("Min distance:", min(distances))
    print("Mean distance:", sum(distances) / len(distances))
    if histogram:
        np_dist = np.array(distances)
        # np_dist = np_dist[np_dist > 30]
        plt.hist(np_dist, bins=40)
        plt.savefig('edge_distance_stats.png')


def get_min_cell_distance(nodes):
    logger.info("Getting the minimum distance between nodes within a frame")
    frame_to_node = {}
    for id_, node in nodes.items():
        frame = node['position'][0]
        coords = node['position'][1:]
        if frame not in frame_to_node:
            frame_to_node[frame] = []
        frame_to_node[frame].append(coords)

    min_distances = {}
    for frame, node_list in frame_to_node.items():
        min_dist = -1
        node_list = sorted(node_list)
        for current_index in range(len(node_list) - 1):
            current = node_list[current_index]
            for compare_to_index in range(current_index + 1, len(node_list)):
                compare_to = node_list[compare_to_index]
                distance = calc_distance(current, compare_to)

                if min_dist == -1 or distance < min_dist:
                    min_dist = distance
                if abs(current[0] - compare_to[0] > min_dist):
                    break
        min_distances[frame] = min_dist
    print("Minimum distance over all frames: %.3f"
          % min(list(min_distances.values())))
    print("Mean minimum distance over frames: %.3f"
          % np.mean(np.array(list(min_distances.values()))))
    logger.debug("Array of minimum distances over frames: %s"
                 % str(min_distances))


def check_valid_edges(nodes, edges, endpoint_names=['source', 'target']):
    logger.info("Checking that each edge points one frame backwards")
    invalid_edges = 0
    for edge in edges:
        source_id = edge[endpoint_names[0]]
        target_id = edge[endpoint_names[1]]
        if target_id == -1 or target_id not in nodes:
            continue
        source_node = nodes[source_id]
        target_node = nodes[target_id]
        source_frame = source_node['position'][0]
        target_frame = target_node['position'][0]
        if target_frame != source_frame - 1:
            invalid_edges += 1
            print("________INVALID EDGE SKIPS FRAMES______________")
            print("Edge:", edge)
            print("Source:", source_node)
            print("Target:", target_node)
    logger.info("Found %d invalid edges" % invalid_edges)


def check_unattached_points(nodes, edges, endpoint_names=['source', 'target']):
    logger.info("Checking that all nodes are attached to an edge")
    id_set = set()
    for edge in edges:
        id_set.add(edge[endpoint_names[0]])
        id_set.add(edge[endpoint_names[1]])

    unattached_nodes = []
    for id in nodes.keys():
        if id not in id_set:
            print("Cell {} not in any edge".format(nodes[id]))
            unattached_nodes.append(id)
    logger.info("Found %d unattached nodes: %s"
                % (len(unattached_nodes), unattached_nodes))
    return unattached_nodes


if __name__ == "__main__":
    path = sys.argv[1]
    nodes, edges = read_nodes_and_edges(path)
    get_edge_distance_stats(nodes, edges)
    check_valid_edges(nodes, edges)
    check_unattached_points(nodes, edges)
    get_min_cell_distance(nodes)
