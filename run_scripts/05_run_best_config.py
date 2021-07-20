from linajea.process_blockwise import solve_blockwise
from linajea import print_time, load_config
from linajea.tracking import TrackingParameters
import linajea.evaluation
from daisy import Roi
import logging
import time
import argparse

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_config', help="test config file")
    parser.add_argument('-vs', '--vald_setup', help="validation setup")
    parser.add_argument('-vr', '--vald_region', help="validation region")
    parser.add_argument('-vf', '--vald_frames', help="validation frames",
                        nargs='+', type=int, default=None)
    parser.add_argument('-vv', '--vald_version',
                        help="limit validation to version")
    parser.add_argument('-vm', '--vald_metric', action='store_true',
                        help="use validation metric (default is sum errors")
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    config = load_config(args.eval_config)
    filter_params = {}
    if args.vald_version:
        filter_params['version'] = args.vald_version
    score_columns = ['validation_score'] if args.vald_metric else None
    best_vald_config = linajea.evaluation.get_best_result(
            args.vald_setup,
            args.vald_region,
            config['general']['db_host'],
            sample=config['general']['sample'],
            frames=args.vald_frames,
            filter_params=filter_params,
            score_columns=score_columns)

    print("Best config: %s" % str(best_vald_config))
    parameters = [TrackingParameters(**best_vald_config)]
    solve_config = config['general']
    solve_keys = config['solve'].keys()
    for key in solve_keys:
        if key == 'num_workers' or key == 'from_scratch':
            continue
        if key in best_vald_config:
            best_value = best_vald_config[key]
            solve_config[key] = best_value

    solve_config['parameters'] = parameters
    print("Solve config: %s" % str(solve_config))
    print("Parameters: %s" % str(parameters))
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
