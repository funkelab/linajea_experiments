from __future__ import absolute_import
from linajea import print_time
from linajea.process_blockwise import predict_blockwise
import logging
import argparse
import time

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
# logging.getLogger('linajea.gunpowder.write_cells').setLevel(logging.DEBUG)
# logging.getLogger('daisy.tasks').setLevel(logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-i', '--iteration', type=int, default=400000)
    args = parser.parse_args()
    config_file = args.config
    iteration = args.iteration
    start_time = time.time()
    predict_blockwise(config_file, iteration)
    end_time = time.time()
    print_time(end_time - start_time)
