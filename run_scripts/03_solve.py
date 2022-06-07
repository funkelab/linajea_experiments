from linajea.process_blockwise import solve_blockwise
from linajea import print_time, load_config, tracking_params_from_config
from daisy import Roi
import linajea.evaluation
import logging
import time
import argparse
import os

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logging.getLogger(
        'linajea.tracking.solver').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="config file")
    parser.add_argument('-pd', '--parameters-dir')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-s', '--solve', action='store_true')
    args = parser.parse_args()

    config_file = args.config
    config = load_config(config_file)
    arguments = config['general']
    arguments.update(config['solve'])
    # arguments['num_workers'] = 5
    # arguments['db_name'] = "linajea_140521_setup11_simple_early_middle_perfect"

    if args.parameters_dir:
        parameters = []
        for config_dir in os.listdir(args.parameters_dir):
            if os.path.isdir(config_dir):
                for config_file in os.listdir(os.path.join(args.parameters_dir, config_dir)):
                    parameters.append(tracking_params_from_config(
                        load_config(os.path.join(args.parameters_dir,
                                                 config_dir,
                                                 config_file))))
            else:
                parameters.append(tracking_params_from_config(
                    load_config(os.path.join(args.parameters_dir,
                                             config_dir))))
    else:
        parameters = [tracking_params_from_config(config)]

    # parameters = [p for p in parameters if p.weight_edge_score != 0.0]

    arguments['parameters'] = parameters

    # check for limit to roi
    predict_config = config['predict']
    if 'limit_to_roi_offset' in predict_config and\
            'limit_to_roi_shape' in predict_config:
        limit_to_roi = Roi(predict_config['limit_to_roi_offset'],
                           predict_config['limit_to_roi_shape'])
        arguments['limit_to_roi'] = limit_to_roi

    start_time = time.time()
    if args.solve:
        solve_blockwise(**arguments)
    print_time(time.time() - start_time)

    if args.evaluate:
        evaluate_args = arguments
        arguments.update(config['evaluate'])
        for parameter in parameters:
            logger.info("%s", str(vars(parameter)))
            evaluate_args['parameters'] = parameter
            start_time = time.time()
            linajea.evaluation.evaluate_setup(**evaluate_args)
            logger.info("Done evaluating for parameters %s", parameter)
            linajea.print_time(time.time() - start_time)
