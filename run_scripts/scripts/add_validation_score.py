from linajea.evaluation import validation_score
from linajea import (
        CandidateDatabase, load_config,
        tracking_params_from_config)
from daisy import Roi
import argparse


def add_validation_score(
        db_name,
        gt_db_name,
        db_host,
        tracking_params,
        frames=None):
    cand_db = CandidateDatabase(db_name,
                                db_host=db_host)
    parameters_id = cand_db.get_parameters_id(
            tracking_params,
            fail_if_not_exists=True)
    cand_db.set_parameters_id(parameters_id)
    total_roi = cand_db.get_nodes_roi()
    if frames:
        total_roi = total_roi.intersect(
                Roi((frames[0], None, None, None),
                    (frames[1] - frames[0], None, None, None)))

    rec_graph = cand_db.get_selected_graph(total_roi)

    gt_db = CandidateDatabase(gt_db_name,
                              db_host=db_host,
                              parameters_id=parameters_id)
    gt_total_roi = gt_db.get_nodes_roi()
    if frames:
        gt_total_roi = gt_total_roi.intersect(
                Roi((frames[0], None, None, None),
                    (frames[1] - frames[0], None, None, None)))
    gt_graph = gt_db.get_graph(gt_total_roi)

    vald_score = validation_score(gt_graph, rec_graph)
    cand_db._MongoDbGraphProvider__connect()
    cand_db._MongoDbGraphProvider__open_db()
    if frames is None:
        score_collection = cand_db.database['scores']
        score_collection.find_one_and_update(
                {'_id': parameters_id},
                {'$set': {'validation_score': vald_score}})
    else:
        score_collection = cand_db.database[
            'scores'+"_".join(str(f) for f in frames)]
        score_collection.find_one_and_update(
                {'param_id': parameters_id},
                {'$set': {'validation_score': vald_score}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    config = load_config(args.config)
    db_name = config['general']['db_name']
    db_host = config['general']['db_host']
    gt_db_name = config['evaluate']['gt_db_name']
    tracking_params = tracking_params_from_config(config)
    frames = config['general']['frames']\
        if 'frames' in config['general'] else None
    add_validation_score(
        db_name,
        gt_db_name,
        db_host,
        tracking_params,
        frames=frames)
