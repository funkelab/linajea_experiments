import linajea
import linajea.tracking
import linajea.evaluation
import logging
import argparse
import time
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def process_config(config):
    evaluate_config = config['general']
    evaluate_config.update(config['solve'])
    evaluate_config.update(config['evaluate'])
    evaluate_config.update({
        'model_type': 'nms',
        'max_cell_move': config['extract_edges']['edge_move_threshold']})

    start_time = time.time()
    linajea.evaluation.evaluate_setup(**evaluate_config)
    logger.info("Done evaluating")
    linajea.print_time(time.time() - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="config file")
    parser.add_argument(
            '-d', '--directory',
            help="process whole directory in a loop")
    args = parser.parse_args()
    if args.directory:
        for f in os.listdir(args.directory):
            config = linajea.load_config(os.path.join(args.directory, f))
            process_config(config)

    else:
        config_file = args.config
        config = linajea.load_config(config_file)
        process_config(config)
