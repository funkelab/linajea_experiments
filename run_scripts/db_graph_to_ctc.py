import argparse
import logging
import os
import sys

import numpy as np

import daisy
import linajea
import linajea.tracking
from linajea.visualization.napari import write_ctc

logger = logging.getLogger(__name__)

logging.basicConfig(level=20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-p', '--param_id', type=int)
    parser.add_argument('--frame_begin', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=None)
    parser.add_argument('--iteration', type=int, default=None,
                        help='checkpoint/iteration to predict')
    parser.add_argument('-g', '--gt', action='store_true')
    parser.add_argument('--sphere', action='store_true')
    parser.add_argument('--setup', type=str,
                        help='path to setup dir')
    parser.add_argument('--sample', type=str, default=None)
    args = parser.parse_args()

    config = linajea.load_config(args.config)
    if args.sample is None:
        sample = config['data']['test_data_dirs'][0]
    else:
        sample = args.sample

    if args.setup is not None:
        config['general']['setup_dir'] = args.setup
    else:
        config['general']['setup_dir'] = os.path.dirname(args.config)

    if args.iteration > 0:
        config['prediction']['iteration'] = args.iteration

    if not args.gt:
        config['general']['db_name'] = linajea.checkOrCreateDB(config, sample)

    data_config = linajea.load_config(
        os.path.join(sample, "data_config.toml"))

    filename_zarr = os.path.join(
        sample, data_config['general']['zarr_file'])

    none_roi = daisy.Roi((None, None, None, None),
                         (None, None, None, None))
    if args.gt:
        logger.info("db {}".format(config['evaluation']['gt_db_name']))
        gt_db = linajea.CandidateDatabase(
            config['evaluation']['gt_db_name'],
            config['general']['db_host'])
        graph = gt_db[none_roi]
    else:
        logger.info("db {}".format(config['general']['db_name']))
        db = linajea.CandidateDatabase(
            config['general']['db_name'],
            config['general']['db_host'],
            parameters_id=args.param_id)
        graph = db.get_selected_graph(none_roi)

    logger.info("Read %d cells and %d edges",
                graph.number_of_nodes(),
                graph.number_of_edges())

    track_graph = linajea.tracking.TrackGraph(
        graph, frame_key='t', roi=graph.roi)

    # if args.frame_begin is not None:
        # assert args.frame_begin >= start_frame, "invalid frame_begin"
    start_frame = args.frame_begin
    # if args.frame_end is not None:
        # assert args.frame_end <= end_frame, "invalid frame_end"
    end_frame = args.frame_end

    if args.gt:
        txt_fn = "man_track.txt"
        tif_fn = "man_track{:03d}.tif"
    else:
        txt_fn = "res_track.txt"
        tif_fn = "mask{:03d}.tif"

    shape = data_config['general']['shape']
    voxel_size = config['data']['voxel_size']
    print("voxel_size", voxel_size)

    write_ctc(track_graph, start_frame, end_frame, shape,
              args.out_dir, txt_fn, tif_fn, voxel_size=voxel_size,
              paint_sphere=args.sphere, gt=args.gt)

if __name__ == "__main__":
    main()
