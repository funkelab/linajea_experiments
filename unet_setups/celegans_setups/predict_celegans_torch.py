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

import h5py
import numpy as np
import gunpowder as gp

from linajea.gunpowder import (WriteCells, Clip,
                               NormalizeMinMax, NormalizeMeanStd,
                               NormalizeMedianMad)
from linajea.process_blockwise import write_done
from linajea.config import TrackingConfig
from linajea import (load_config,
                     construct_zarr_filename)

import torch
import torch_model

# try:
#     import absl.logging
#     logging.root.removeHandler(absl.logging._absl_handler)
#     absl.logging._warn_preinit_stderr = False
# except Exception as e:
#     print(e)

# wrong crop: uncomment
# logging.basicConfig(
#         level=logging.DEBUG,
#         format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


logger = logging.getLogger(__name__)


def predict_sample(config):

    raw = gp.ArrayKey('RAW')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    if not config.model.train_only_cell_indicator:
        parent_vectors = gp.ArrayKey('PARENT_VECTORS')

    model = torch_model.UnetModelWrapper(config, config.inference.checkpoint)
    model.eval()
    print(model)

    input_shape = config.model.predict_input_shape
    trial_run = model.forward(torch.zeros(input_shape, dtype=torch.float32))
    _, _, trial_max, _ = trial_run
    output_shape = trial_max.size()

    voxel_size = gp.Coordinate(config.inference.data_source.voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    # wrong crop: comment parent_vectors (and maxima)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)
    if not config.model.train_only_cell_indicator:
        chunk_request.add(parent_vectors, output_size)

    sample = config.inference.data_source.datafile.filename
    if os.path.isdir(sample):
        data_config = load_config(
            os.path.join(sample, "data_config.toml"))
        filename_zarr = os.path.join(
            sample, data_config['general']['zarr_file'])
    else:
        data_config = load_config(
            os.path.join(os.path.dirname(sample), "data_config.toml"))
        filename_zarr = os.path.join(
            os.path.dirname(sample), data_config['general']['zarr_file'])
    source = gp.ZarrSource(
        filename_zarr,
        datasets={
            raw: config.inference.data_source.datafile.group
        },
        nested="nested" in config.inference.data_source.datafile.group,
        array_specs={
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size)})

    if os.path.isdir(sample):
        filename_mask = os.path.join(
            sample,
            data_config['general'].get(
                'mask_file',
                os.path.splitext(data_config['general']['zarr_file'])[0] + "_mask.hdf"))
    else:
        filename_mask = sample + "_mask.hdf"

    with h5py.File(filename_mask, 'r') as f:
        mask = np.array(f['volumes/mask'])
    z_range = data_config['general']['z_range']
    if z_range[1] < 0:
        z_range[1] = data_config['general']['shape'][1] - z_range[1]

    with gp.build(source):
        raw_spec = source.spec[raw]

    logger.info("raw spec %s", raw_spec)

    if config.train.augment.normalization is None or \
       config.train.augment.normalization == 'default':
        logger.info("default normalization")
        source = source + \
            gp.Normalize(raw,
                         factor=1.0/np.iinfo(data_config['stats']['dtype']).max)
    elif config.train.augment.normalization == 'minmax':
        mn = config.train.augment.norm_bounds[0]
        mx = config.train.augment.norm_bounds[1]
        logger.info("minmax normalization %s %s", mn, mx)
        source = source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx)
    elif config.train.augment.normalization == 'percminmax':
        mn = data_config['stats'][config.train.augment.perc_min]
        mx = data_config['stats'][config.train.augment.perc_max]
        logger.info("perc minmax normalization %s %s", mn, mx)
        source = source + \
            Clip(raw, mn=mn/2, mx=mx*2) + \
            NormalizeMinMax(raw, mn=mn, mx=mx)
    elif config.train.augment.normalization == 'mean':
        mean = data_config['stats']['mean']
        std = data_config['stats']['std']
        mn = data_config['stats'][config.train.augment.perc_min]
        mx = data_config['stats'][config.train.augment.perc_max]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        source = source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMeanStd(raw, mean=mean, std=std)
    elif config.train.augment.normalization == 'median':
        median = data_config['stats']['median']
        mad = data_config['stats']['mad']
        mn = data_config['stats'][config.train.augment.perc_min]
        mx = data_config['stats'][config.train.augment.perc_max]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        source = source + \
            Clip(raw, mn=mn, mx=mx) + \
            NormalizeMedianMad(raw, median=median, mad=mad)
    else:
        raise RuntimeError("invalid normalization method %s",
                           config.train.augment.normalization)

    inputs={
        'raw': raw
    }
    outputs={
        0: cell_indicator,
        1: maxima,
        # 2: raw_cropped,
    }
    if not config.model.train_only_cell_indicator:
        outputs[3] = parent_vectors


    dataset_names={
        # wrong crop: comment parent_vectors (and maxima)
        cell_indicator: 'volumes/cell_indicator',
        maxima: '/volumes/maxima',
    }
    if not config.model.train_only_cell_indicator:
        dataset_names[parent_vectors] = 'volumes/parent_vectors'

    pipeline = (
        source +
        gp.Pad(raw, size=None) +
        gp.torch.Predict(
            model=model,
            checkpoint=os.path.join(config.general.setup_dir,
                                    'train_net_checkpoint_{}'.format(
                                        config.inference.checkpoint)),
            inputs=inputs,
            outputs=outputs,
            use_swa=config.predict.use_swa,
            # skip_empty=True
        ))

    cb = []
    if config.predict.write_to_zarr:
        pipeline = (
            pipeline +

            gp.ZarrWrite(
                dataset_names=dataset_names,
                output_filename=construct_zarr_filename(config, sample,
                                                        config.inference.checkpoint)
            ))
        # wrong crop: (optionally) comment
        if not config.predict.no_db_access:
            cb.append(lambda b: write_done(
                b,
                'predict_zarr',
                config.inference.data_source.db_name,
                config.general.db_host))
        else:
            cb.append(lambda _: True)

    if config.predict.write_to_db:
        pipeline = (
            pipeline +

            WriteCells(
                maxima,
                cell_indicator,
                parent_vectors if not config.model.train_only_cell_indicator else None,
                score_threshold=config.inference.cell_score_threshold,
                db_host=config.general.db_host,
                db_name=config.inference.data_source.db_name,
                mask=mask,
                z_range=z_range,
                volume_shape=data_config['general']['shape'])
            )
        cb.append(lambda b: write_done(
            b,
            'predict_db',
            db_name=config.inference.data_source.db_name,
            db_host=config.general.db_host))


    roi_map={
        raw: 'read_roi',
        # wrong crop: comment parent_vectors (and maxima)
        cell_indicator: 'write_roi',
        maxima: 'write_roi'
    }
    if not config.model.train_only_cell_indicator:
        roi_map[parent_vectors] = 'write_roi'

    pipeline = (
        pipeline +

        gp.PrintProfilingStats(every=100) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map=roi_map,
            num_workers=1,
            block_done_callback=lambda b, st, et: all([f(b) for f in cb])
    ))

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    predict_sample(config)