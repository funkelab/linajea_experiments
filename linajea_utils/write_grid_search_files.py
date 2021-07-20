import argparse
from linajea import load_config
import json
import toml
import itertools
import shutil
import os
import logging

logger = logging.getLogger(__name__)


def write_configs(out_dir, base_config, values, keys,
                  ignore_existing=False, json_format=False):
    existing_configs = []
    if not ignore_existing:
        config_num = len(os.listdir(out_dir)) + 1
        for fl in os.listdir(out_dir):
            with open(os.path.join(out_dir, fl), 'r') as f:
                existing_configs.append(json.load(f))
    else:
        config_num = 1

    for config_vals in values:
        logger.debug("Config vals %s" % str(config_vals))
        whole_config_copy = base_config.copy()
        solve_config = whole_config_copy['solve']
        for index, key in enumerate(keys):
            solve_config[key] = config_vals[index]
        whole_config_copy['solve'] = solve_config
        if whole_config_copy in existing_configs:
            logger.warn("skipping config {}, already exists".format(config_vals))
            continue
        logger.debug("writing config {}, num: {}".format(config_vals, config_num))
        with open(os.path.join(out_dir, str(config_num)), 'w') as outfile:
            if json_format:
                json.dump(whole_config_copy, outfile, indent=2)
            else:
                toml.dump(whole_config_copy, outfile)
        config_num += 1

    return config_num - 1


def write_grid_search_configs(
        sample_config,
        out_dir,
        grid_params,
        ignore_existing=False,
        frame_start=None,
        frame_end=None,
        json_format=False,
        group_size=None):

    logger.info("Writing config files from %s to %s"
                % (sample_config, out_dir))

    if (frame_start is not None and frame_end is None) or \
       (frame_start is None and frame_end is not None):
        logger.warn("Warning: both frame_start and frame_end have to be set"
                    " (or neither), ignoring...")
    config = load_config(sample_config)
    grid_search = load_config(grid_params)
    logger.debug("Loaded config files")

    if ignore_existing and os.path.exists(out_dir):
        logger.debug("Removing existing contents of output directory")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_search_keys = list(grid_search.keys())

    product = list(itertools.product(*[grid_search[key]
                                     for key in grid_search_keys]))

    if frame_start is not None and frame_end is not None:
        config['general']['frames'] = [frame_start,
                                       frame_end]

    if group_size:
        for i in range(0, len(product), group_size):
            group_dir = os.path.join(out_dir, str(i // group_size))
            os.makedirs(group_dir, exist_ok=True)
            group_vals = product[i: min(i+group_size, len(product))]
            write_configs(group_dir, config, group_vals, grid_search_keys,
                          ignore_existing=ignore_existing,
                          json_format=json_format)
    else:
        write_configs(out_dir, config, product, grid_search_keys,
                      ignore_existing=ignore_existing,
                      json_format=json_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='sample config file')
    parser.add_argument('--output', type=str, required=True,
                        help='output dir')
    parser.add_argument('--grid_params', type=str, required=True,
                        help='grid search params json')
    parser.add_argument('--ignore_existing', action="store_true",
                        help='remove output dir if exists')
    parser.set_defaults(check_existing=None)
    parser.add_argument('--frame_start', type=int, default=None,
                        help='start frame (start and end frame or neither)')
    parser.add_argument('--frame_end', type=int, default=None,
                        help='end frame (start and end frame or neither)')
    parser.add_argument('--json', action='store_true',
                        help='Format to write config files. Default is toml')
    args = parser.parse_args()
    write_grid_search_configs(
            args.config,
            args.output,
            args.grid_params,
            ignore_existing=args.ignore_existing,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            json=args.json)
