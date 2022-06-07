from linajea import CandidateDatabase, load_config
from linajea.tracking import TrackGraph
from linajea.evaluation import match_nodes
from daisy import Roi
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_best_effort(
        db_name,
        gt_db_name,
        db_host,
        roi,
        matching_threshold,
        best_effort_key,
        ):
    cand_db = CandidateDatabase(db_name, db_host, 'a')
    gt_db = CandidateDatabase(gt_db_name, db_host, 'r')
    cand_graph = cand_db[roi]
    gt_graph = gt_db[roi]

    cand_track_graph = TrackGraph(
        cand_graph, frame_key='t', roi=roi)
    gt_track_graph = TrackGraph(
        gt_graph, frame_key='t', roi=roi)

    logger.debug("Starting matching")
    node_matches = match_nodes(
            cand_track_graph,
            gt_track_graph,
            matching_threshold)

    gt_to_cand = {}

    logger.info("Found %d matches", len(node_matches))

    # update best effort key in local graph
    for cand_id, gt_id in node_matches:
        cand_graph.nodes[cand_id][best_effort_key] = True
        gt_to_cand[gt_id] = cand_id

    for gt_u, gt_v in gt_graph.edges():
        if gt_u not in gt_to_cand or gt_v not in gt_to_cand:
            # one of the endpoints didn't have a match!
            continue
        rec_u = gt_to_cand[gt_u]
        rec_v = gt_to_cand[gt_v]
        if rec_v in cand_graph[rec_u]:
            assert rec_v in cand_graph[rec_u],\
                'Assumed there was an edge between the rec matches'\
                ' for the endpoints of a given gt edge'
            cand_graph[rec_u][rec_v][best_effort_key] = True

    # save best effort key to database
    cand_graph.update_node_attrs(
            roi,
            attributes=[best_effort_key])
    cand_graph.update_edge_attrs(
            roi,
            attributes=[best_effort_key])
    logger.info("Done updating node and edge attributes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-k', '--best_effort_key', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    db_name = config['db_name']
    gt_db_name = config['gt_db_name']
    db_host = config['db_host']
    offset = config['offset']
    shape = config['shape']
    best_effort_key = args.best_effort_key
    if best_effort_key is None:
        assert 'best_effort_key' in config, "must specify best effort key"
        best_effort_key = config['best_effort_key']
    roi = Roi(offset, shape)
    matching_threshold = config['matching_threshold']
    get_best_effort(
        db_name,
        gt_db_name,
        db_host,
        roi,
        matching_threshold,
        best_effort_key,
        )
