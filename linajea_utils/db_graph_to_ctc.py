import argparse
import logging
import os
import h5py

import numpy as np

import daisy
import linajea
import linajea.tracking
from linajea.visualization.ctc import write_ctc

logger = logging.getLogger(__name__)

logging.basicConfig(level=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-p', '--param_id')
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--frame_begin', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=None)
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='checkpoint/iteration to predict')
    parser.add_argument('-g', '--gt', action='store_true')
    parser.add_argument('--sphere', action='store_true')
    parser.add_argument('--ws', type=str, default=None)
    parser.add_argument('--fg_threshold', type=float, default=0.5)
    parser.add_argument('--setup', type=str,
                        help='path to setup dir')
    parser.add_argument('--sample', type=str, default=None)
    args = parser.parse_args()
    print(args)

    config = linajea.load_config(args.config)

    if args.setup is not None:
        config['general']['setup_dir'] = args.setup
    else:
        config['general']['setup_dir'] = os.path.dirname(args.config)

    start_frame = args.frame_begin
    end_frame = args.frame_end
    roi = daisy.Roi((start_frame, None, None, None),
                    (end_frame - start_frame, None, None, None))
    if args.gt:
        logger.info("db {}".format(config['evaluation']['gt_db_name']))
        gt_db = linajea.CandidateDatabase(
            config['evaluation']['gt_db_name'],
            config['general']['db_host'])
        graph = gt_db[roi]
    else:
        param_id = args.param_id
        try:
            param_id = int(param_id)
            logger.info("db {}".format(config['general']['db_name']))
            db = linajea.CandidateDatabase(
                config['general']['db_name'],
                config['general']['db_host'],
                parameters_id=args.param_id)
            graph = db.get_selected_graph(roi)
        except:
            selected_key = param_id
            logger.info("db {}".format(config['general']['db_name']))
            db = linajea.CandidateDatabase(
                config['general']['db_name'],
                config['general']['db_host'])
            db.selected_key = selected_key
            graph = db.get_selected_graph(roi)

    logger.info("Read %d cells and %d edges",
                graph.number_of_nodes(),
                graph.number_of_edges())

    track_graph = linajea.tracking.TrackGraph(
        graph, frame_key='t', roi=graph.roi)

    if args.gt:
        txt_fn = "man_track.txt"
        tif_fn = "man_track{:03d}.tif"
    else:
        txt_fn = "res_track.txt"
        tif_fn = "mask{:03d}.tif"

    voxel_size = config['general']['voxel_size']
    print("voxel_size", voxel_size)

    if args.ws is not None:
        dataset = daisy.open_ds(args.ws, 'volumes/cell_indicator')
        raw_dataset = daisy.open_ds(
                '../data/ctc_drosophila/' + config['general']['data_file'], 'raw')
        raw_roi = raw_dataset.roi
        ws_roi = raw_roi.intersect(dataset.roi)
        surface = dataset[ws_roi].to_ndarray()
        shape = surface.shape
        if 'mask_filename' in config['general']:
            filename_mask = config['general']['mask_file']
            with h5py.File(filename_mask, 'r') as f:
                mask = np.array(f['volumes/mask'])
                mask = np.reshape(mask, (1, ) + mask.shape)
        else:
            mask = None
    else:
        surface = None
        mask = None

    write_ctc(track_graph, start_frame, end_frame, shape,
              args.out_dir, txt_fn, tif_fn, voxel_size=voxel_size,
              paint_sphere=args.sphere, gt=args.gt, surface=surface,
              fg_threshold=args.fg_threshold, mask=mask)
