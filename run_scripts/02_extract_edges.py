from __future__ import absolute_import
import argparse
import logging
import time

from linajea import print_time, load_config
from linajea.process_blockwise import extract_edges_blockwise


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        help='path to config file')
    args = parser.parse_args()
    config = load_config(args.config_file)

    start_time = time.time()
    extract_edges_config = config['general']
    extract_edges_config.update(config['extract_edges'])
    extract_edges_config['frame_context'] = config['solve']['context'][0]
    start_time = time.time()
    extract_edges_blockwise(**extract_edges_config)
    print_time(time.time() - start_time)
