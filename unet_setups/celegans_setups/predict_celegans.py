import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)

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

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)


def predict_sample(config, sample, setup_dir):
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    net_config = load_config(os.path.join(setup_dir, 'test_net_config.json'))

    raw = gp.ArrayKey('RAW')
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')

    voxel_size = gp.Coordinate(config['data']['voxel_size'])
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape_2'])*voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(parent_vectors, output_size)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)

    data_config = load_config(os.path.join(sample, "data_config.toml"))
    filename_zarr = os.path.join(sample, data_config['general']['zarr_file'])
    source = gp.ZarrSource(
        filename_zarr,
        datasets={
            raw: 'volumes/raw'},
        array_specs={
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size)})

    with gp.build(source):
        raw_spec = source.spec[raw]

    logger.info("raw spec %s", raw_spec)
    if 'normalization' not in config['training'] or \
       config['training']['normalization'] == 'default':
        logger.info("default normalization")
        source = source + \
                 gp.Normalize(raw, factor=1.0/(256*256-1))
    elif config['training']['normalization'] == 'minmax':
        mn = config['training']['norm_min']
        mx = config['training']['norm_max']
        logger.info("minmax normalization %s %s", mn, mx)
        source = source + \
                 Clip(raw, mn=mn/2, mx=mx*2) + \
                 NormalizeMinMax(raw, mn=mn, mx=mx)
    elif config['training']['normalization'] == 'mean':
        mean = data_config['stats']['mean']
        std = data_config['stats']['std']
        mn = data_config['stats'][config['training']['perc_min']]
        mx = data_config['stats'][config['training']['perc_max']]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        source = source + \
                 Clip(raw, mn=mn, mx=mx) + \
                 NormalizeMeanStd(raw, mean=mean, std=std)
    elif config['training']['normalization'] == 'median':
        median = data_config['stats']['median']
        mad = data_config['stats']['mad']
        mn = data_config['stats'][config['training']['perc_min']]
        mx = data_config['stats'][config['training']['perc_max']]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        source = source + \
                 Clip(raw, mn=mn, mx=mx) + \
                 NormalizeMedianMad(raw, median=median, mad=mad)
    else:
        raise RuntimeError("invalid normalization method %s",
                           config['training']['normalization'])
    pipeline = (
        source +
        gp.Pad(raw, size=None) +
        gp.tensorflow.Predict(
            os.path.join(setup_dir,
                         'train_net_checkpoint_{}'.format(
                             config['prediction']['iteration'])),
            graph=os.path.join(setup_dir, 'test_net.meta'),
            inputs={
                net_config['raw']: raw,
            },
            outputs={
                net_config['parent_vectors_cropped']: parent_vectors,
                net_config['cell_indicator_cropped']: cell_indicator,
                net_config['maxima']: maxima,
            },
            skip_empty=True
        ))

    cb = []
    if config['prediction']['write_zarr']:
        pipeline = (
            pipeline +

            gp.ZarrWrite(
                dataset_names={
                    parent_vectors: 'volumes/parent_vectors',
                    cell_indicator: 'volumes/cell_indicator',
                    maxima: '/volumes/maxima',
                },
                output_filename=construct_zarr_filename(config, sample)
            ))
        cb.append(lambda b: write_done(
            b,
            'predict_zarr',
            config['general']['db_name'],
            config['general']['db_host']))

    if config['prediction']['write_cells_db']:
        pipeline = (
            pipeline +

            WriteCells(
                maxima,
                cell_indicator,
                parent_vectors,
                score_threshold=config['prediction']['cell_score_threshold'],
                db_host=config['general']['db_host'],
                db_name=config['general']['db_name'],
                volume_shape=data_config['general']['shape'])
            )
        cb.append(lambda b: write_done(
            b,
            'predict_cells',
            config['general']['db_name'],
            config['general']['db_host']))

    pipeline = (
        pipeline +

        gp.PrintProfilingStats(every=100) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
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
    os.makedirs(args.setup_dir, exist_ok=True)
    if args.db:
        config['general']['db_name'] = args.db

    predict_sample(config, args.sample, args.setup_dir)
