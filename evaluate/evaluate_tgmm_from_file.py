from linajea import CandidateDatabase
from linajea.tracking import TrackGraph
from linajea.evaluation import evaluate
from daisy import Roi
import sys
import logging
from networkx import DiGraph
import time
import os
import pymongo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tgmm_graph(tracks_file, frames=None, scale=True):

    nodes = []
    edges = []

    for line in open(tracks_file, 'r'):

        tokens = line.split()
        #   0  1  2  3  4        5          6         7
        try:
            t, z, y, x, cell_id, parent_id, track_id, _ = tokens
        except:
            t, z, y, x, cell_id, parent_id, track_id = tokens
        t = int(float(t))
        if frames is not None:
            if t < frames[0] or t > frames[1]:
                continue
        z = float(z)
        if scale:
            z = z * 5
        y = float(y)
        x = float(x)
        parent_id = int(parent_id)
        cell_id = int(cell_id)
        track_id = int(track_id)

        nodes.append(
            (cell_id,
             {'t': t,
              'z': z,
              'y': y,
              'x': x,
              'id': cell_id,
              'score': 0}
             ))

        if parent_id >= 0:
            edges.append(
                (cell_id,
                 parent_id,
                 {'score': 0,
                  'distance': 0}
                 ))

    if len(nodes) == 0:
        logger.error("Did not find any nodes in file %s" % tracks_file)
        return
    logger.info("Found %s nodes and %s edges" % (len(nodes), len(edges)))
    subgraph = DiGraph()

    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)
    logger.info("Added %s nodes and %s edges"
                % (len(subgraph.nodes), len(subgraph.edges)))
    if subgraph.number_of_edges() == 0:
        logger.warn("No tgmm edges in file %s" % tracks_file)
    return subgraph


def evaluate_tgmm(tgmm_graph, gt_db_name, db_host, matching_threshold,
                  frames=None, error_details=False):
    source_roi = Roi((0, 0, 0, 0), (10e7, 10e7, 10e7, 10e7))
    # limit to specific frames, if given
    if frames:
        begin, end = frames
        crop_roi = Roi(
            (begin, None, None, None),
            (end - begin, None, None, None))
        source_roi = source_roi.intersect(crop_roi)

    logger.info("Evaluating in %s", source_roi)

    if tgmm_graph.number_of_edges() == 0:
        return

    tgmm_track_graph = TrackGraph(
        tgmm_graph, frame_key='t')

    gt_db = CandidateDatabase(gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db[source_roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))
    gt_track_graph = TrackGraph(
        gt_subgraph, frame_key='t')

    logger.info("Matching edges for for tgmm and gt")
    score = evaluate(
            gt_track_graph,
            tgmm_track_graph,
            matching_threshold=matching_threshold,
            sparse=True)

    logger.info("Done evaluating results for tgmm")
    return score


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_gt_to_mongo.py"
              " <tracks_file>")
        sys.exit(1)
    tracks_file = sys.argv[1]
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea",
    gt_db_name = 'linajea_120828_gt'
    matching_threshold = 15
    frames = [0, 300]
    tgmm_graph = get_tgmm_graph(tracks_file, frames=frames)
    score = evaluate_tgmm(tgmm_graph, gt_db_name, db_host, matching_threshold, frames=frames)
    fname = os.path.basename(tracks_file)
    tgmm_id = os.path.splitext(fname)[0]
    logger.info("Done evaluating results for %s. Saving results to mongo."
                % tgmm_id)
    tgmm_db_name = 'linajea_120828_tgmm_scores'
    client = pymongo.MongoClient(db_host)
    tgmm_scores_db = client[tgmm_db_name]
    score_collection = tgmm_scores_db['scores']
    eval_dict = {'_id': tgmm_id}
    eval_dict.update(score.__dict__)
    score_collection.replace_one({'_id': tgmm_id},
                                 eval_dict,
                                 upsert=True)
