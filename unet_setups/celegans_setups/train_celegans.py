from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import json
import time
import toml
import zarr
from linajea.gunpowder import (TracksSource, AddParentVectors,
                               ShiftAugment, Clip, NoOp,
                               NormalizeMinMax, NormalizeMeanStd,
                               NormalizeMedianMad)
from linajea import (load_config,
                     parse_limit_roi)

import gunpowder as gp
import logging
import math
import os
import sys
import tensorflow as tf
import numpy as np

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)


def merge_sources(
        source,
        tracks,
        center_tracks,
        track_file,
        center_cell_file,
        file_roi,
        ):
    return (
        # raw
        (source,
         # tracks
         TracksSource(
             track_file,
             tracks,
             points_spec=gp.PointsSpec(roi=file_roi)),
         # center tracks
         TracksSource(
             center_cell_file,
             center_tracks,
             points_spec=gp.PointsSpec(roi=file_roi)),
         ) +
        gp.MergeProvider() +
        gp.Pad(tracks, None) +
        gp.Pad(center_tracks, None))


def train_until(config, setup_dir):
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    # Get the latest checkpoint.
    if tf.train.latest_checkpoint(setup_dir):
        trained_until = int(tf.train.latest_checkpoint(
            setup_dir).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= config['training']['iterations']:
        return

    net_config = load_config(os.path.join(setup_dir, 'train_net_config.json'))

    raw = gp.ArrayKey('RAW')
    tracks = gp.PointsKey('TRACKS')
    center_tracks = gp.PointsKey('CENTER_TRACKS')
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    cell_mask = gp.ArrayKey('CELL_MASK')
    pred_parent_vectors = gp.ArrayKey('PRED_PARENT_VECTORS')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    grad_parent_vectors = gp.ArrayKey('GRAD_PARENT_VECTORS')
    grad_cell_indicator = gp.ArrayKey('GRAD_CELL_INDICATOR')

    voxel_size = gp.Coordinate(config['data']['voxel_size'])
    input_size = gp.Coordinate(
            net_config['input_shape']) * voxel_size
    output_size_1 = gp.Coordinate(
            net_config['output_shape_1']) * voxel_size
    output_size_2 = gp.Coordinate(
            net_config['output_shape_2']) * voxel_size
    center = [max(1, int(d * 0.75)) for d in net_config['output_shape_1']]
    center_size = gp.Coordinate(center) * voxel_size
    # add a buffer in time to avoid choosing a random location based on points
    # in only one frame, because then all points are rejected as being on the
    # lower boundary of that frame
    center_size = center_size + gp.Coordinate((1, 0, 0, 0))
    logger.info("Center size: {}".format(center_size))
    logger.info("Output size 1: {}".format(output_size_1))

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(tracks, output_size_1)
    request.add(center_tracks, center_size)
    request.add(parent_vectors, output_size_1)
    request.add(cell_indicator, output_size_1)
    request.add(cell_mask, output_size_1)
    logger.info("REQUEST: %s" % str(request))
    snapshot_request = gp.BatchRequest({
        raw: request[raw],
        pred_parent_vectors: request[parent_vectors],
        pred_cell_indicator: request[parent_vectors],
        grad_parent_vectors: request[parent_vectors],
        grad_cell_indicator: request[parent_vectors]
    })
    snapshot_request.add(maxima, output_size_2)
    logger.info("Snapshot request: %s" % str(snapshot_request))

    num_files = len(config['data']['train_data_dirs'])
    sources = []
    for t in range(num_files):
        d = config['data']['train_data_dirs'][t]
        logger.info("loading data %s", d)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_zarr = os.path.join(
            d, data_config['general']['zarr_file'])
        filename_tracks = os.path.join(
            d, data_config['general']['tracks_file'])
        filename_daughters = os.path.join(
            d, data_config['general']['daughter_cells_file'])
        logger.info("creating source: %s (%s, %s), divisions?: %s",
                    filename_zarr, filename_tracks, filename_daughters,
                    config['training']['divisions'])
        file_size = data_config['general']['shape']
        file_roi = gp.Roi(
            gp.Coordinate((0, 0, 0, 0)),
            gp.Coordinate(file_size)) * voxel_size
        logger.info("file roi: %s", file_roi)
        limit_to_roi = parse_limit_roi(config['training'])
        if limit_to_roi is not None:
            logger.info("limit to roi: %s", limit_to_roi)
            file_roi = file_roi.intersect(limit_to_roi)
            logger.info("roi (intersection): %s", file_roi)
        file_source = gp.ZarrSource(
            filename_zarr,
            datasets={
                raw: 'volumes/raw'},
            array_specs={
                raw: gp.ArraySpec(
                    roi=file_roi,
                    interpolatable=True,
                    voxel_size=voxel_size)})
        if 'normalization' not in config['training'] or \
           config['training']['normalization'] == 'default':
            logger.info("default normalization")
            file_source = file_source + \
                          gp.Normalize(raw, factor=1.0/(256*256-1))
        elif config['training']['normalization'] == 'minmax':
            mn = config['training']['norm_min']
            mx = config['training']['norm_max']
            logger.info("minmax normalization %s %s", mn, mx)
            file_source = file_source + \
                          Clip(raw, mn=mn/2, mx=mx*2) + \
                          NormalizeMinMax(raw, mn=mn, mx=mx)
        elif config['training']['normalization'] == 'mean':
            mean = data_config['stats']['mean']
            std = data_config['stats']['std']
            mn = data_config['stats'][config['training']['perc_min']]
            mx = data_config['stats'][config['training']['perc_max']]
            logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
            file_source = file_source + \
                          Clip(raw, mn=mn, mx=mx) + \
                          NormalizeMeanStd(raw, mean=mean, std=std)
        elif config['training']['normalization'] == 'median':
            median = data_config['stats']['median']
            mad = data_config['stats']['mad']
            mn = data_config['stats'][config['training']['perc_min']]
            mx = data_config['stats'][config['training']['perc_max']]
            logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
            file_source = file_source + \
                          Clip(raw, mn=mn, mx=mx) + \
                          NormalizeMedianMad(raw, median=median, mad=mad)
        else:
            raise RuntimeError("invalid normalization method %s",
                               config['training']['normalization'])

        source = merge_sources(
            file_source,
            tracks,
            center_tracks,
            filename_tracks,
            filename_tracks,
            file_roi)
        if config['training']['divisions']:
            div_source = merge_sources(
                file_source,
                tracks,
                center_tracks,
                filename_tracks,
                filename_daughter,
                file_roi)

            source = (source, div_source) + \
                gp.RandomProvider(probabilities=[0.75, 0.25])
        source = source + \
                 gp.Pad(raw, gp.Coordinate((0, 80, 55, 55))) + \
                 gp.RandomLocation(
                     ensure_nonempty=center_tracks,
                     p_nonempty=config['training']['reject_empty_prob'])
        sources.append(source)

    augmentation = kwargs['training']['augmentation']
    pipeline = (
        tuple(sources) +
        gp.RandomProvider() +

        (gp.ElasticAugment(
            augmentation['elastic']['control_point_spacing'],
            augmentation['elastic']['jitter_sigma'],
            [augmentation['elastic']['rotation_min']*np.pi/180.0,
             augmentation['elastic']['rotation_max']*np.pi/180.0],
            subsample=augmentation['elastic'].get('subsample', 1),
            spatial_dims=3) \
         if augmentation.get('elastic') is not None else NoOp()) +
        (ShiftAugment(
            prob_slip=augmentation['shift']['prob_slip'],
            prob_shift=augmentation['shift']['prob_shift'],
            sigma=augmentation['shift']['sigma'],
            shift_axis=0) \
         if augmentation.get('shift') is not None else NoOp()) +
        (gp.SimpleAugment(
            mirror_only=augmentation['simple'].get("mirror"),
            transpose_only=augmentation['simple'].get("transpose")) \
         if augmentation.get('simple') is not None else NoOp()) +

        (gp.IntensityAugment(
            raw,
            scale_min=augmentation['intensity']['scale'][0],
            scale_max=augmentation['intensity']['scale'][1],
            shift_min=augmentation['intensity']['shift'][0],
            shift_max=augmentation['intensity']['shift'][1],
            z_section_wise=False) \
         if augmentation.get('intensity') is not None else NoOp()) +

        AddParentVectors(
            tracks,
            parent_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            radius=config['training']['parent_radius'],
            move_radius=config['data']['cell_move_radius'],
            dense=config['training']['dense_gt']) +

        gp.RasterizePoints(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=config['training']['rasterize_radius'],
                mode='peak')) +

        (gp.Reject(
            ensure_nonempty=tracks,
            mask=cell_mask,
            min_masked=0.0001,
            reject_probability=0.9) \
         if not kwargs['training']['dense_gt'] else NoOp()) +

        (gp.PreCache(
            cache_size=kwargs['training']['cache_size'],
            num_workers=kwargs['training']['num_workers']) \
         if not kwargs['general']['debug'] else NoOp()) +

        gp.tensorflow.Train(
            os.path.join(setup_dir, 'train_net'),
            optimizer=net_config['optimizer'],
            loss=net_config['loss'],
            summary=net_config['summary'],
            inputs={
                net_config['raw']: raw,
                net_config['gt_parent_vectors']: parent_vectors,
                net_config['gt_cell_indicator']: cell_indicator,
                net_config['cell_mask']: cell_mask,
            },
            outputs={
                net_config['parent_vectors']: pred_parent_vectors,
                net_config['cell_indicator']: pred_cell_indicator,
                net_config['maxima']: maxima,
            },
            gradients={
                net_config['parent_vectors']: grad_parent_vectors,
                net_config['cell_indicator']: grad_cell_indicator,
            },
            log_dir=setup_dir,
            save_every=config['training']['checkpoints']) +

        # visualize
        gp.IntensityScaleShift(raw, scale=100.0, shift=0) +
        gp.Snapshot({
                raw: 'volumes/raw',
                parent_vectors: 'volumes/parent_vectors',
                cell_indicator: 'volumes/cell_indicator',
                cell_mask: 'volumes/cell_mask',
                pred_parent_vectors: 'volumes/pred_parent_vectors',
                pred_cell_indicator: 'volumes/pred_cell_indicator',
                maxima: 'volumes/maxima',
                grad_parent_vectors: 'volumes/grad_parent_vectors',
                grad_cell_indicator: 'volumes/grad_cell_indicator',
            },
            output_dir=os.path.join(setup_dir, 'snapshots'),
            output_filename='snapshot_{iteration}.hdf',
            additional_request=snapshot_request,
            every=config['training']['snapshots'],
                dataset_dtypes={
                    maxima: np.uint8
            }) +
        gp.PrintProfilingStats(every=config['training']['profiling'])
    )

    with gp.build(pipeline):

        logger.info("Starting training...")
        for i in range(trained_until, config['training']['iterations']):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--iterations', type=int, default=-1,
                        help='number ofiterations to train')
    parser.add_argument('--setup_dir', type=str,
                        required=True, help='output')

    args = parser.parse_args()
    config = load_config(args.config)
    if args.iterations > 0:
        config['training']['iterations'] = args.iterations
    os.makedirs(setup_dir, exist_ok=True)

    train_until(config, args.setup_dir)
