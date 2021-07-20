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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("iteration", type=int)
    args = parser.parse_args()
    start_time = time.time()
    predict_blockwise(args.config_file, args.iteration)
    end_time = time.time()
    print_time(end_time - start_time)
