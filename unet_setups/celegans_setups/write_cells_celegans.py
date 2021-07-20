import argparse
import logging
import os
import sys

import numpy as np
import gunpowder as gp

from linajea.gunpowder import (WriteCells, Clip,
                               NormalizeMinMax, NormalizeMeanStd,
                               NormalizeMedianMad)
from linajea.process_blockwise import write_done
from linajea import (load_config,
                     construct_zarr_filename)

from util_celegans import checkOrCreateDB

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)


def write_cells_sample(config, sample, setup_dir):
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    net_config = load_config(os.path.join(setup_dir, 'test_net_config.json'))

    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')

    voxel_size = gp.Coordinate(config['data']['voxel_size'])
    output_size = gp.Coordinate(net_config['output_shape_2'])*voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(parent_vectors, output_size)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)

    data_config = load_config(os.path.join(sample, "data_config.toml"))
    output_path = construct_zarr_filename(config, sample)
    source = gp.ZarrSource(
        output_path,
        datasets={
            parent_vectors: 'volumes/parent_vectors',
            cell_indicator: 'volumes/cell_indicator',
            maxima: '/volumes/maxima'},
        array_specs={
            parent_vectors: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size),
            cell_indicator: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size),
            maxima: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)})

    cb = []
    if config['prediction']['write_cells_db']:
        cb.append(lambda b: write_done(
            b,
            'predict_cells',
            config['general']['db_name'],
            config['general']['db_host']))

    pipeline = (
        source +
        gp.Pad(parent_vectors, size=None) +
        gp.Pad(cell_indicator, size=None) +
        gp.Pad(maxima, size=None) +

        WriteCells(
            maxima,
            cell_indicator,
            parent_vectors,
            score_threshold=config['prediction']['cell_score_threshold'],
            db_host=config['general']['db_host'],
            db_name=config['general']['db_name'],
            volume_shape=data_config['general']['shape']) +
        gp.PrintProfilingStats(every=100) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                parent_vectors: 'write_roi',
                cell_indicator: 'write_roi',
                maxima: 'write_roi'
            },
            num_workers=5,
            block_done_callback=lambda b, st, et: all([f(b) for f in cb])
    ))

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--sample', type=str,
                        help='path to sample')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='which saved model to use for prediction')
    parser.add_argument('--setup_dir', type=str,
                        required=True, help='output')
    parser.add_argument('--db', type=str, help='db name')

    args = parser.parse_args()

    config = load_config(args.config)
    if args.iteration > 0:
        config['prediction']['iteration'] = args.iteration
    os.makedirs(setup_dir, exist_ok=True)
    if args.db:
        config['general']['db_name'] = args.db

    write_cells_sample(config, args.sample, args.setup_dir)
