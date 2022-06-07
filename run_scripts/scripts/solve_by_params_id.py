from linajea.process_blockwise import solve_blockwise
from linajea import print_time, load_config, CandidateDatabase
from linajea.tracking import TrackingParameters
import linajea.evaluation
from daisy import Roi
import logging
import time
import argparse
import sys

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
# logging.getLogger(
#         'daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_config', help="target setup config file")
    parser.add_argument('-vdb', '--vald_db', help="validation database", required=True)
    parser.add_argument('-p', '--params_id', help="parameter id in validation database", required=True, type=int)
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    config = load_config(args.eval_config)
    db_host = config['general']['db_host']
    vald_db = linajea.CandidateDatabase(args.vald_db, db_host)
    vald_parameters = vald_db.get_parameters(args.params_id)
    if vald_parameters is None:
        print("No parameters found in db %s with id %s"
              % (args.vald_db, args.params_id))
        sys.exit()

    tracking_params = TrackingParameters(**vald_parameters)
    parameters = [tracking_params]
    solve_config = config['general']
    solve_keys = config['solve'].keys()
    for key in solve_keys:
        if key == 'num_workers' or key == 'from_scratch':
            continue
        if key in tracking_params.__dict__:
            best_value = tracking_params.__dict__[key]
            solve_config[key] = best_value

    solve_config['parameters'] = parameters
    print("Solve config: %s" % str(solve_config))
    print("Parameters: %s" % str(tracking_params.__dict__))
    # check for limit to roi
    predict_config = config['predict']
    if 'limit_to_roi_offset' in predict_config and\
            'limit_to_roi_shape' in predict_config:
        limit_to_roi = Roi(predict_config['limit_to_roi_offset'],
                           predict_config['limit_to_roi_shape'])
        solve_config['limit_to_roi'] = limit_to_roi

    start_time = time.time()
    solve_blockwise(**solve_config)
    print_time(time.time() - start_time)

    if args.evaluate:
        evaluate_config = solve_config
        evaluate_config.update(config['evaluate'])
        start_time = time.time()
        linajea.evaluation.evaluate_setup(**evaluate_config)
        logger.info("Done evaluating")
        linajea.print_time(time.time() - start_time)
