from __future__ import absolute_import
from linajea import print_time, load_config
from linajea.process_blockwise import extract_edges_blockwise
import logging
import sys
import time

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


if __name__ == "__main__":

    config_file = sys.argv[1]
    config = load_config(config_file)

    extract_edges_config = config['general']
    extract_edges_config.update(config['extract_edges'])
    extract_edges_config['frame_context'] = config['solve']['context'][0]
    start_time = time.time()
    extract_edges_blockwise(**extract_edges_config)
    print_time(time.time() - start_time)
