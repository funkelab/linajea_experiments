import linajea
import linajea.tracking
import linajea.evaluation
import time
import daisy
import pymongo
import logging
import numpy as np
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_tgmm_score(
        results_db_name,
        mongo_url,
        score,
        start_frame=None,
        end_frame=None):
    if start_frame is None:
        collection_name = 'scores'
    else:
        collection_name = 'scores' + str(start_frame) + '_' + str(end_frame)
    client = pymongo.MongoClient(mongo_url)
    results_db = client[results_db_name]
    collection = results_db.get_collection(collection_name)
    id_name = 'tgmm'
    eval_dict = {'_id': id_name}
    eval_dict.update(score.__dict__)
    for key, value in eval_dict.items():
        if type(value) == np.uint64:
            eval_dict[key] = int(value)
    collection.replace_one({'_id': id_name}, eval_dict, upsert=True)
    return eval_dict


def evaluate_setups(
        tgmm_db_name,
        gt_db_name,
        matching_threshold,
        start_frame=None,
        end_frame=None,
        results_db_name=None):
    if start_frame:
        assert(end_frame)
        assert(start_frame < end_frame)
        num_frames = end_frame - start_frame
        roi = daisy.Roi((start_frame, 0, 0, 0), (num_frames, 1e10, 1e10, 1e10))
    else:
        roi = daisy.Roi((0, 0, 0, 0), (1e10, 1e10, 1e10, 1e10))

    mongo_url = "localhost"  # TODO: Replace with MongoDB URL
    tgmm_db = linajea.CandidateDatabase(tgmm_db_name, mongo_url)
    gt_db = linajea.CandidateDatabase(gt_db_name, mongo_url)

    logger.info("Reading GT cells and edges in %s" % roi)
    gt_subgraph = gt_db[roi]
    gt_graph = linajea.tracking.TrackGraph(gt_subgraph, frame_key='t')
    logger.info("Found %d nodes and %d edges in GT"
                % (gt_graph.number_of_nodes(), gt_graph.number_of_edges()))
    logger.info("Reading cells and edges in %s" % roi)
    tgmm_subgraph = tgmm_db[roi]
    tgmm_graph = linajea.tracking.TrackGraph(tgmm_subgraph, frame_key='t')
    logger.info("Found %d nodes and %d edges in tgmm"
                % (tgmm_graph.number_of_nodes(), tgmm_graph.number_of_edges()))

    logger.info("Evaluating tgmm")
    score = linajea.evaluation.evaluate(
            gt_graph,
            tgmm_graph,
            matching_threshold=matching_threshold,
            sparse=True)
    logger.info(score)
    logger.info("Writing to mongo")
    if not results_db_name:
        results_db_name = tgmm_db_name
    write_tgmm_score(results_db_name, mongo_url, score,
                     start_frame=start_frame, end_frame=end_frame)
    logger.info("Done writing to mongo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tgmm_db_name')
    parser.add_argument('gt_db_name')
    parser.add_argument('matching_threshold', type=int)
    parser.add_argument('-s', '--start-frame', type=int, default=None)
    parser.add_argument('-e', '--end-frame', type=int, default=None)
    parser.add_argument('-r', '--results-db-name', default=None)
    args = parser.parse_args()

    start_time = time.time()
    evaluate_setups(
            args.tgmm_db_name,
            args.gt_db_name,
            args.matching_threshold,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            results_db_name=args.results_db_name)
    end_time = time.time()
    linajea.print_time(end_time - start_time)
