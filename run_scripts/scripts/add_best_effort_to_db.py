import argparse
import logging
import os

import pymongo

from linajea import (CandidateDatabase,
                     checkOrCreateDB)
import daisy
from linajea.config import TrackingConfig

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--db_name', type=str, default=None,
                        help='db_name')
    parser.add_argument('--edge_collection', type=str, default="edges",
                        help='name of edge collection in db')
    parser.add_argument('--gt_db_name', type=str, default=None,
                        help='gt db_name')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='checkpoint')
    parser.add_argument('--dry_run', action="store_true")
    parser.add_argument('--non_minimal', action="store_true")
    parser.add_argument('--param_id', type=int, default=0,
                        help=("id to be used in db: selected_ID, default 0 "
                              "(usually not used)"))
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')

    args = parser.parse_args()
    config = TrackingConfig.from_file(os.path.abspath(args.config))

    roi = daisy.Roi(offset=[0, 0, 0, 0], shape=[9999, 9999, 9999, 9999])

    if args.validation:
        roi = config.validate_data.data_sources[0].roi
        sample = config.validate_data.data_sources[0]
        checkpoint = config.validate_data.checkpoints[0]
        cell_score_threshold = config.validate_data.cell_score_threshold
    else:
        roi = config.test_data.data_sources[0].roi
        sample = config.test_data.data_sources[0]
        checkpoint = config.test_data.checkpoint
        cell_score_threshold = config.test_data.cell_score_threshold
    if config.evaluate.parameters.roi is not None:
        roi = config.evaluate.parameters.roi
    if args.checkpoint is not None:
        checkpoint = args.checkpoint

    roi_d = daisy.Roi(offset=roi.offset, shape=roi.shape)

    if args.db_name is None:
        db_name = checkOrCreateDB(
            config.general.db_host,
            config.general.setup_dir,
            sample.datafile.filename,
            checkpoint,
            cell_score_threshold,
            tag=config.general.tag)
    else:
        db_name = args.db_name
    if args.gt_db_name is None:
        gt_db_name = sample.gt_db_name
    else:
        gt_db_name = args.gt_db_name

    db_graph = CandidateDatabase(db_name, config.general.db_host,
                                 mode='r')[roi_d]
    gt_db_graph = CandidateDatabase(gt_db_name, config.general.db_host,
                                    mode='r')[roi_d]

    client = pymongo.MongoClient(host=config.general.db_host)
    db = client[db_name]
    edges_db = db[args.edge_collection]

    selected_key = "selected_" + str(args.param_id)
    # backwards -> source in later frame
    cnt_edges_t = 0
    cnt_edges_f = 0
    for source, target in db_graph.edges():
        query = {"source": int(source),
                 "target": int(target)}
        if "probable_gt_cell_id" in db_graph.nodes()[source] and \
           "probable_gt_cell_id" in db_graph.nodes()[target]:
            source_prob_gt_cell_id = db_graph.nodes()[source][
                "probable_gt_cell_id"]
            target_prob_gt_cell_id = db_graph.nodes()[target][
                "probable_gt_cell_id"]
            if (source_prob_gt_cell_id, target_prob_gt_cell_id) in \
               gt_db_graph.edges():
                update = {"$set": {selected_key: True}}
                cnt_edges_t += 1
            else:
                update = {"$set": {selected_key: False}}
                cnt_edges_f += 1
        else:
            update = {"$set": {selected_key: False}}
            cnt_edges_f += 1
        logger.debug("updating %s with %s", query, update)
        if not args.dry_run:
            edges_db.update_one(query, update)
    logger.info("updated %d edges (true: %d, false: %d",
                cnt_edges_t+cnt_edges_f, cnt_edges_t, cnt_edges_f)

    parameters_db = db["parameters"]
    parameters = {
        "_id": args.param_id,
        "block_size": [-1, -1, -1, -1],
        "context": [-1, -1, -1, -1],
        "roi": {
            "offset": roi.offset,
            "shape": roi.shape
        },
    }
    if args.non_minimal:
        parameters.update({"cost_appear": -1,
                           "cost_disappear": -1,
                           "cost_split": -1,
                           "max_cell_move": -1,
                           "threshold_node_score": -1,
                           "weight_node_score": -1,
                           "threshold_edge_score": -1,
                           "weight_prediction_distance_cost": -1,
                           "use_cell_state": False,
                           "threshold_split_score": -1,
                           "threshold_is_normal_score": -1,
                           "threshold_is_daughter_score": -1,
                           "cost_daughter": -1,
                           "cost_normal": -1,
                           "use_cell_cycle_indicator": False
                           })
    else:
        parameters.update(
            {"weight_node_score": -1,
             "selection_constant": -1,
             "track_cost": -1,
             "weight_division": -1,
             "division_constant": -1,
             "weight_child": -1,
             "weight_continuation": -1,
             "weight_edge_score": -1,
             "cell_cycle_key": config.solve.parameters[0].cell_cycle_key
             if config.solve.parameters[0].cell_cycle_key is not None else "",
             "max_cell_move": -1
             })

    logger.info("inserting parameters %s", parameters)
    if not args.dry_run:
        try:
            parameters_db.insert_one(parameters)
            logger.info("updated parameters collection")
        except pymongo.errors.DuplicateKeyError:
            logger.info("parameters already in collection")
