import linajea
from linajea.tracking import TrackGraph
import daisy
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_for_duplicate_gt_tracks(
        gt_db_name,
        db_host,
        start_frame,
        end_frame,
        cell_radius,
        node_overlap_threshold):
    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 10000, 10000, 10000))
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)
    gt_graph = gt_db[roi]
    if not gt_graph.number_of_edges():
        logger.info("No edges in database. Skipping track formation.")
        return

    track_graph = TrackGraph(gt_graph)
    tracks = track_graph.get_tracks()

    logger.info("Found {} tracks".format(len(tracks)))
    for track_id, track in enumerate(tracks):
        for cell_id in track.nodes():
            track_graph.nodes[cell_id]['track_id'] = track_id

    # Count how many times each pair of tracks has cells close to each other
    dup_counts = {}
    for frame, cells in track_graph._cells_by_frame.items():
        for i in range(len(cells)):
            cell1 = cells[i]
            for j in range(i + 1, len(cells)):
                cell2 = cells[j]
                if 'track_id' not in track_graph.nodes[cell1]:
                    print(cell1)
                    print(track_graph.nodes[cell1])
                track1 = track_graph.nodes[cell1]['track_id']
                track2 = track_graph.nodes[cell2]['track_id']
                if track1 == track2:
                    continue
                if close(track_graph.nodes[cell1],
                         track_graph.nodes[cell2],
                         cell_radius):
                    logger.info("These cells are close together: %d %s    %d %s"
                                % (cell1, track_graph.nodes[cell1],
                                   cell2, track_graph.nodes[cell2]))
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


def check_missing_edges(
        gt_db_name,
        db_host,
        start_frame,
        end_frame,
        move_distance):
    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 10000, 10000, 10000))
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)
    gt_graph = gt_db[roi]
    if not gt_graph.number_of_edges():
        logger.info("No edges in database. Skipping track formation.")
        return

    track_graph = TrackGraph(gt_graph)
    tracks = track_graph.get_tracks()

    logger.info("Found {} tracks".format(len(tracks)))
    for track_id, track in enumerate(tracks):
        for cell_id in track.nodes():
            track_graph.nodes[cell_id]['track_id'] = track_id

    missing_edges = {}
    for frame, cells in track_graph._cells_by_frame.items():
        prev_frame = frame - 1
        if prev_frame not in track_graph._cells_by_frame:
            continue
        prev_cells = track_graph._cells_by_frame[prev_frame]
        for i in range(len(cells)):
            cell1 = cells[i]
            if len(track_graph.prev_edges(cell1)) > 0:
                continue
            for j in range(len(prev_cells)):
                cell2 = prev_cells[j]
                if 'track_id' not in track_graph.nodes[cell1]:
                    print(cell1)
                    print(track_graph.nodes[cell1])
                track1 = track_graph.nodes[cell1]['track_id']
                track2 = track_graph.nodes[cell2]['track_id']
                if track1 == track2:
                    continue
                if close(track_graph.nodes[cell1],
                         track_graph.nodes[cell2],
                         move_distance):
                    logger.info("These cells are close together and "
                                "potentially missing an edge between them: "
                                "%d %s    %d %s"
                                % (cell1, track_graph.nodes[cell1],
                                   cell2, track_graph.nodes[cell2]))
                    missing_edges[(cell1, cell2)] = (track_graph.nodes[cell1],
                                                     track_graph.nodes[cell2])

    print("Found %d potential missing edges:" % len(missing_edges))
    approved_ids = []
    for ids, locs in missing_edges.items():
        print("%s %s %s %s" % (ids[0], locs[0], ids[1], locs[1]))
        merge = input().strip()
        while merge not in ['y', 'n']:
            merge = input("Please enter y or n")
        if merge == 'y':
            approved_ids.append(ids)
    print("Approved %d missing edges: %s" % (len(approved_ids), approved_ids))


def check_degree(
        gt_db_name,
        db_host,
        start_frame,
        end_frame):
    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 10000, 10000, 10000))
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)
    logger.info("Reading GT cells and edges in %s" % roi)
    gt_subgraph = gt_db[roi]
    node_degrees = {node: degree for node, degree in gt_subgraph.in_degree()}
    max_node_degree = max(node_degrees.values())
    logger.info("Max node degree for subgraph: %s" % max_node_degree)
    if max_node_degree > 2:
        logger.info("Expected max in_degree <=2, got %d. \n %s"
                    % (max_node_degree, str(node_degrees)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_db_name")
    parser.add_argument("cell_radius")
    parser.add_argument("node_overlap_threshold")
    parser.add_argument("move_threshold")
    parser.add_argument("-f", "--frames", type=int, nargs=2, default=[0, 10e5])
    args = parser.parse_args()
    gt_db_name = args.gt_db_name
    start_frame, end_frame = args.frames
    cell_radius = args.cell_radius
    node_overlap_threshold = args.node_overlap_threshold
    move_threshold = args.move_threshold

    db_host = "localhost"

    check_for_duplicate_gt_tracks(
            gt_db_name, db_host,
            start_frame, end_frame,
            int(cell_radius), int(node_overlap_threshold))
    check_missing_edges(
            gt_db_name, db_host,
            start_frame, end_frame,
            int(move_threshold))
    check_degree(gt_db_name, db_host, start_frame, end_frame)
