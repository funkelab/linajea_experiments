import argparse
import logging
from linajea.evaluation.validation_metric import validation_score
from linajea import CandidateDatabase
from daisy import Roi


logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger.setLevel(logging.DEBUG)


def get_validation_score(
        gt_db,
        rec_db,
        key,
        host):
    # get nx graphs containing lineages
    logger.debug("Loading lineages")
    gt_cand_db = CandidateDatabase(gt_db, host)
    rec_cand_db = CandidateDatabase(rec_db, host, parameters_id=key)
    total_roi = Roi((225, 0, 0, 0), (50, 10e8, 10e8, 10e8))
    gt_lineages = gt_cand_db[total_roi]
    logger.info("%d nodes in gt", len(gt_lineages))
    rec_lineages = rec_cand_db.get_selected_graph(total_roi)

    # compute validation score
    logger.debug("Computing validation score")
    vald_score = validation_score(gt_lineages, rec_lineages)

    logger.info("Vald score for %s with key %d and gt %s: %f"
                % (rec_db, key, gt_db, vald_score))
    return vald_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-db')
    parser.add_argument('--rec_db')
    parser.add_argument('--key', type=int)
    args = parser.parse_args()
    host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea",
    get_validation_score(args.gt_db, args.rec_db, args.key, host)
