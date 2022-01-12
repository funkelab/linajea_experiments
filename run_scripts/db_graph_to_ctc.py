import argparse
import logging
import os

import numpy as np
import h5py
import zarr

import daisy
import linajea
import linajea.tracking
from linajea.visualization.ctc import write_ctc

logger = logging.getLogger(__name__)

logging.basicConfig(level=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-p', '--param_id', type=int)
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
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

    for config in linajea.getNextInferenceData(args):
        data = config.inference.data_source
        fn = data.datafile.filename
        if args.sample is not None:
            if fn != args.sample:
                if os.path.dirname(fn) != args.sample:
                    continue
                else:
                    fn = os.path.dirname(fn)

        data_config = linajea.load_config(os.path.join(fn, "data_config.toml"))

        if config.evaluate.parameters.roi is not None:
            assert config.evaluate.parameters.roi.shape[0] <= data.roi.shape[0], \
                "your evaluation ROI is larger than your data roi!"
            data.roi = config.evaluate.parameters.roi
        else:
            config.evaluate.parameters.roi = data.roi
        evaluate_roi = daisy.Roi(offset=data.roi.offset,
                                 shape=data.roi.shape)

        if args.gt:
            logger.info("db {}".format(data.gt_db_name))
            gt_db = linajea.CandidateDatabase(
                data.gt_db_name,
                config.general.db_host)
            graph = gt_db[evaluate_roi]
        else:
            logger.info("db {}".format(data.db_name))
            db = linajea.CandidateDatabase(
                data.db_name,
                config.general.db_host,
                parameters_id=args.param_id)
            graph = db.get_selected_graph(evaluate_roi)

            if config.evaluate.parameters.filter_polar_bodies or \
               config.evaluate.parameters.filter_polar_bodies_key:
                if not config.evaluate.parameters.filter_polar_bodies and \
                   config.evaluate.parameters.filter_polar_bodies_key is not None:
                    pb_key = config.evaluate.parameters.filter_polar_bodies_key
                else:
                    pb_key = config.solve.parameters[0].cell_cycle_key + "polar"
                tmp_subgraph = db.get_selected_graph(evaluate_roi)
                for node in list(tmp_subgraph.nodes()):
                    if tmp_subgraph.degree(node) > 2:
                        es = list(tmp_subgraph.predecessors(node))
                        tmp_subgraph.remove_edge(es[0], node)
                        tmp_subgraph.remove_edge(es[1], node)
                rec_graph = linajea.tracking.TrackGraph(
                    tmp_subgraph, frame_key='t', roi=tmp_subgraph.roi)

                for track in rec_graph.get_tracks():
                    cnt_nodes = 0
                    cnt_polar = 0
                    for node_id, node in track.nodes(data=True):
                        cnt_nodes += 1
                        try:
                            if node[pb_key] > 0.5:
                                cnt_polar += 1
                        except KeyError:
                            pass
                    if cnt_polar/cnt_nodes > 0.5:
                        graph.remove_nodes_from(track.nodes())
                        logger.info("removing %s potential polar nodes", len(track.nodes()))

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

        shape = data_config['general']['shape']
        voxel_size = data.voxel_size

        if args.ws is not None:
            surface = np.array(zarr.open(args.ws, 'r')[
                'volumes/cell_indicator'][:args.frame_end])
            filename_mask = os.path.join(fn,
                                         data_config['general']['mask_file'])
            with h5py.File(filename_mask, 'r') as f:
                mask = np.array(f['volumes/mask'])
                mask = np.reshape(mask, (1, ) + mask.shape)
        else:
            surface = None
            mask = None

        write_ctc(track_graph, args.frame_begin, args.frame_end, shape,
                  args.out_dir, txt_fn, tif_fn, voxel_size=voxel_size,
                  paint_sphere=args.sphere, gt=args.gt, surface=surface,
                  fg_threshold=args.fg_threshold, mask=mask)

        print("stopping after first data sample")
        break


if __name__ == "__main__":
    main()
