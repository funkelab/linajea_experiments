from linajea import load_config
import linajea.evaluation
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
    parser.add_argument('-vdb', '--vald_db', help="validation database")
    parser.add_argument('-vf', '--vald_frames', help="validation frames",
                        nargs='+', type=int, default=None)
    parser.add_argument('-vv', '--vald_version',
                        help="limit validation to version")
    parser.add_argument('-vm', '--vald_metric', action='store_true',
                        help="use validation metric (default is sum errors")
    args = parser.parse_args()

    config = load_config(args.eval_config)
    filter_params = {}
    if args.vald_version:
        filter_params['version'] = args.vald_version
    score_columns = ['validation_score'] if args.vald_metric else None
    best_vald_config = linajea.evaluation.get_best_result(
            args.vald_db,
            config['general']['db_host'],
            sample=config['general']['sample'],
            frames=args.vald_frames,
            filter_params=filter_params, score_columns=score_columns)

    print("Best config: %s" % str(best_vald_config))
    solve_config = config['general']
    solve_keys = config['solve'].keys()
    for key in solve_keys:
        if key == 'num_workers' or key == 'from_scratch':
            continue
        if key in best_vald_config:
            best_value = best_vald_config[key]
            solve_config[key] = best_value

    evaluate_config = solve_config
    evaluate_config.update(config['evaluate'])
    print("Eval config: %s" % str(evaluate_config))
    start_time = time.time()
    linajea.evaluation.evaluate_setup(**evaluate_config)
    logger.info("Done evaluating")
    linajea.print_time(time.time() - start_time)
