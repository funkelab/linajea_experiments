import argparse
from linajea_utils import write_grid_search_configs
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='sample config file')
    parser.add_argument('grid_params', type=str,
                        help='grid search params json')
    parser.add_argument('output', type=str,
                        help='output dir')
    parser.add_argument('--ignore_existing', action="store_true",
                        help='remove output dir if exists')
    parser.set_defaults(check_existing=None)
    parser.add_argument('--frame_start', type=int, default=None,
                        help='start frame (start and end frame or neither)')
    parser.add_argument('--frame_end', type=int, default=None,
                        help='end frame (start and end frame or neither)')
    parser.add_argument('--json', action='store_true',
                        help='Format to write config files. Default is toml')
    parser.add_argument(
            '--group_size', type=int, default=None,
            help="Write configs into subdirectories with this many configs in each")
    args = parser.parse_args()
    write_grid_search_configs(
            args.config,
            args.output,
            args.grid_params,
            ignore_existing=args.ignore_existing,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            json_format=args.json,
            group_size=args.group_size)
