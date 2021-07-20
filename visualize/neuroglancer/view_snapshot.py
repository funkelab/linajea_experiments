import neuroglancer as ng
import h5py
import os
import numpy as np
import logging
import argparse
from linajea import load_config
import sys

logging.basicConfig(level=logging.WARN)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# TODO: redirect print statements from neuroglancer service to file
# (instead of printing them over top of the prompt)


def get_layer_args(volume, name, voxel_size, threeD_shader, color=None):
    volume = np.array(volume, dtype=np.float32)
    logger.info("Shape of {}: {}".format(
        name,
        volume.shape))
    args = {'name': name}
    if len(volume.shape) == 5:
        logger.debug("Original shape of {}: {}".format(
            name,
            volume.shape))
        volume = volume[:, 0, :, :, :]
        args['shader'] = threeD_shader
    elif color == 0:
        args['shader'] = """
                     void main() {
                       emitRGB(vec3(toNormalized(getDataValue()),0,0));
                     }
                     """
    elif color == 2:
        args['shader'] = """
                     void main() {
                       emitRGB(vec3(toNormalized(0,getDataValue()),0));
                     }
                     """
    offset = tuple(-1 * s * v / 2
                   for s, v in zip(volume.shape[:-4:-1], voxel_size))

    args['layer'] = ng.LocalVolume(
                        data=volume,
                        offset=offset,
                        voxel_size=voxel_size)
    return args


def print_help():
    print(
        """
        USAGE:
        Type any number at the prompt to view that snapshot.
        Enter an empty line to skip to the next snapshot (according to skip_by)
        or, if batch_size is specified, the next sample in a batch
        Other options:
            "new setup" (or "new", "n"): switch to a new setup
            "new config" (or "config", "c"): switch to a new config file
            "exit": exit the program
            "help" (or "h"): display this help
        """)


def load_snapshot(path, config):
    snapshot = h5py.File(path, 'r')
    logger.debug("Available keys: {}".format(list(snapshot.keys())))
    volumes = snapshot['volumes']
    logger.info("Available volumes: {}".format(list(volumes.keys())))

    raw = volumes['raw']
    logger.info("Shape of raw: {}".format(raw.shape))

    to_add = []
    for name in config['layers']:
        if name.startswith('raw'):
            indices = [int(i) for i in name.split('_')[1:]]
            data = raw
            for i in indices:
                data = data[i]
            color = indices[-1] - 2
            to_add.append(get_layer_args(
                data, name,
                config['voxel_size'],
                config['threeD_shader'],
                color=color))
        else:
            if name not in volumes:
                print("No data with name %s" % name)
                continue
            data = volumes[name]
            to_add.append(get_layer_args(
                data, name,
                config['voxel_size'],
                config['threeD_shader'],
                ))

    ng.set_server_bind_address(bind_address='0.0.0.0', bind_port=0)
    viewer = ng.Viewer()

    with viewer.txn() as s:
        s.voxel_size = config['voxel_size']
        for args in to_add:
            s.layers.append(**args)

    return viewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--snap_file', type=str,
                        help='path to snap file')
    parser.add_argument('--setup', type=str,
                        help="name of setup folder")
    parser.add_argument('--snapshot-num', type=int,
                        help='snapshot to start at')
    parser.add_argument('--skip-by', type=int,
                        help='skip by this number when automatically'
                        ' reading next snapshot')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='set logging to debug (default is warn)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.snap_file:
        viewer = load_snapshot(args.snap_file, config)
        print(viewer)
        print("enter to exit")
        input()
        sys.exit(0)


    snapshot_num = None
    if args.setup:
        setup = args.setup
    elif 'setup' in config:
        setup = config['setup']
    else:
        setup = input("Setup name: ")

    skip_by = args.skip_by if args.skip_by else int(input("Skip by: "))
    batch_index = 0

    while(True):
        if snapshot_num is None:
            print_help()
        if snapshot_num is None and args.snapshot_num:
            print("Starting with snapshot %d" % args.snapshot_num)
            _input = args.snapshot_num
        else:
            _input = input(">>: ")

        if _input == 'exit':
            break
        elif _input == 'help' or _input == 'h':
            print_help()
            continue
        elif _input == 'new setup' or\
                _input == 'new' or\
                _input == 'n':
            print("Switching to new setup")
            setup = input("Setup name: ")
            continue
        elif _input == '':
            if 'batch_size' in config:
                batch_index += 1
                if batch_index >= config['batch_size']:
                    print("skipping to next batch")
                    batch_index = 0
                    snapshot_num += skip_by
                else:
                    print("skipping to next sample in batch "
                          "(%d / %d) % batch_index)"
                          % (batch_index, config['batch_size']))
            else:
                snapshot_num += skip_by
                print("skipping to next snapshot at {}".format(snapshot_num))
        else:
            try:
                snapshot_num = int(_input)
            except ValueError:
                print("input not an integer")
                print_help()
                continue

        base_path = config['base_path']
        snapshot_path = config['snapshot_path']\
            if 'snapshot_path' in config else "snapshots"
        snapshot_pattern = config['snapshot_pattern']\
            if 'snapshot_path' in config else 'snapshot_%d.hdf'
        setup_path = os.path.join(config['base_path'], setup, snapshot_path)

        path = os.path.join(setup_path, snapshot_pattern % snapshot_num)
        logger.debug("Loading snapshot at path %s" % path)
        if not os.path.isfile(path):
            print("No file at path %s. Try again" % path)
            continue

        snapshot = h5py.File(path, 'r')
        logger.debug("Available keys: {}".format(list(snapshot.keys())))
        volumes = snapshot['volumes']
        logger.info("Available volumes: {}".format(list(volumes.keys())))

        raw = volumes['raw']
        logger.info("Shape of raw: {}".format(raw.shape))

        to_add = []
        for name in config['layers']:
            if name.startswith('raw') and '_' in name:
                time_index = int(name.split('_')[1])
                print(raw.shape)
                if 'batch_size' in config:
                    data = raw[batch_index][time_index]
                else:
                    data = raw[time_index]
                color = time_index - 2
                to_add.append(get_layer_args(
                    data, name,
                    config['voxel_size'],
                    config['threeD_shader'],
                    color=color))
            else:
                if name not in volumes:
                    print("No data with name %s" % name)
                    continue
                data = volumes[name]
                if 'batch_size' in config:
                    data = data[batch_index]
                if len(data.shape) < 3:
                    print("%s has values %s" % (name, data))
                else:
                    to_add.append(get_layer_args(
                        data, name,
                        config['voxel_size'],
                        config['threeD_shader'],
                        ))

        ng.set_server_bind_address(bind_address='0.0.0.0', bind_port=0)
        viewer = ng.Viewer()

        with viewer.txn() as s:
            s.voxel_size = config['voxel_size']
            for args in to_add:
                s.layers.append(**args)

        print(viewer)
