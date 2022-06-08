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
from linajea.config import TrackingConfig
from linajea import load_config

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


logger = logging.getLogger(__name__)

def merge_sources(
        raw,
        tracks,
        center_tracks,
        track_file,
        center_cell_file,
        limit_to_roi,
        scale=1.0,
        use_radius=False
        ):
    return (
        (raw,
         # tracks
         TracksSource(
             track_file,
             tracks,
             points_spec=gp.PointsSpec(roi=limit_to_roi),
             scale=scale,
             use_radius=use_radius),
         # center tracks
         TracksSource(
             center_cell_file,
             center_tracks,
             points_spec=gp.PointsSpec(roi=limit_to_roi),
             scale=scale,
             use_radius=use_radius),
         ) +
        gp.MergeProvider() +
        # not None padding works in combination with ensure_nonempty in
        # random_location as always a random point is picked and the roi
        # shifted such that that point is inside
        gp.Pad(tracks, gp.Coordinate((0, 500, 500, 500))) +
        gp.Pad(center_tracks, gp.Coordinate((0, 500, 500, 500)))
        # gp.Pad(tracks, None) +
        # gp.Pad(center_tracks, None)
    )


def train_until(config):
    # Get the latest checkpoint.
    if tf.train.latest_checkpoint(config.general.setup_dir):
        trained_until = int(tf.train.latest_checkpoint(
            config.general.setup_dir).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= config.train.max_iterations:
        return

    net_config = load_config(os.path.join(config.general.setup_dir,
                                          'train_net_config.json'))
    net_names = load_config(os.path.join(config.general.setup_dir,
                                         'train_net_names.json'))
    if 'summaries' in net_names:
        summaries = net_names['summaries']
    else:
        summaries = None
        for k, v in net_names.items():
            if "summaries" in k:
                if summaries is None:
                    summaries = {}
                if "scalar" in k:
                    summaries[k] = (v, 1)
                    if config.train.val_log_step is not None:
                        summaries["validation_" + k] = (v, config.train.val_log_step)
                elif "network" in k:
                    summaries[k] = (v, 25)
                elif "histo" in k:
                    summaries[k] = (v, 50)
                elif "image" in k:
                    summaries[k] = (v, 50)
                elif "metric" in k:
                    summaries[k] = (v, 10)
                    if config.train.val_log_step is not None:
                        summaries["validation_" + k] = (v, config.train.val_log_step)
                else:
                    summaries[k] = (v, 10)


    anchor = gp.ArrayKey('ANCHOR')
    raw = gp.ArrayKey('RAW')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    tracks = gp.PointsKey('TRACKS')
    center_tracks = gp.PointsKey('CENTER_TRACKS')
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    cell_center = gp.ArrayKey('CELL_CENTER')
    cell_mask = gp.ArrayKey('CELL_MASK')
    pred_parent_vectors = gp.ArrayKey('PRED_PARENT_VECTORS')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    grad_parent_vectors = gp.ArrayKey('GRAD_PARENT_VECTORS')
    grad_cell_indicator = gp.ArrayKey('GRAD_CELL_INDICATOR')
    if config.train.cell_density:
        cell_density = gp.ArrayKey('CELL_DENSITY')
        pred_cell_density = gp.ArrayKey('PRED_CELL_DENSITY')
    else:
        cell_density = None

    voxel_size = gp.Coordinate(config.train_data.data_sources[0].voxel_size)
    input_size = gp.Coordinate(
            net_config['input_shape']) * voxel_size
    output_size_1 = gp.Coordinate(
            net_config['output_shape_1']) * voxel_size
    output_size_2 = gp.Coordinate(
            net_config['output_shape_2']) * voxel_size
    # center = [max(1, int(d * 0.75)) for d in net_config['output_shape_2']]
    center_size = gp.Coordinate(net_config['output_shape_2']) * voxel_size
    # add a buffer in time to avoid choosing a random location based on points
    # in only one frame, because then all points are rejected as being on the
    # lower boundary of that frame
    center_size = center_size + gp.Coordinate((1, 0, 0, 0))
    logger.info("Center size: {}".format(center_size))
    logger.info("Output size 1: {}".format(output_size_1))
    logger.info("Voxel size: {}".format(voxel_size))

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(tracks, output_size_1)
    request.add(center_tracks, center_size)
    request.add(parent_vectors, output_size_1)
    request.add(cell_indicator, output_size_1)
    request.add(raw_cropped, output_size_1)
    request.add(cell_center, output_size_1)
    request.add(cell_mask, output_size_1)
    request.add(anchor, output_size_1)
    logger.info("REQUEST: %s" % str(request))
    snapshot_request = gp.BatchRequest({
        raw: request[raw],
        raw_cropped: request[parent_vectors],
        pred_parent_vectors: request[parent_vectors],
        pred_cell_indicator: request[parent_vectors],
        # grad_parent_vectors: request[parent_vectors],
        # grad_cell_indicator: request[parent_vectors]
    })
    snapshot_request.add(maxima, output_size_2)
    if config.train.cell_density:
        request.add(cell_density, output_size_1)
        snapshot_request.add(cell_density, output_size_1)
        snapshot_request.add(pred_cell_density, output_size_1)
    logger.info("Snapshot request: %s" % str(snapshot_request))


    train_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                config.train_data.data_sources,
                                cell_density=cell_density)
    if config.train.val_log_step is not None:
        val_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                  config.validate_data.data_sources,
                                  cell_density=cell_density)

    augment = config.train.augment
    train_pipeline = (
        tuple(train_sources) +
        gp.RandomProvider() +

        (gp.ElasticAugment(
            augment.elastic.control_point_spacing,
            augment.elastic.jitter_sigma,
            [augment.elastic.rotation_min*np.pi/180.0,
             augment.elastic.rotation_max*np.pi/180.0],
            subsample=augment.elastic.subsample,
            use_fast_points_transform=augment.elastic.use_fast_points_transform,
            spatial_dims=3,
            temporal_dim=True) \
         if augment.elastic is not None else NoOp()) +

        (ShiftAugment(
            prob_slip=augment.shift.prob_slip,
            prob_shift=augment.shift.prob_shift,
            sigma=augment.shift.sigma,
            shift_axis=0) \
         if augment.shift is not None else NoOp()) +

        (gp.SimpleAugment(
            mirror_only=augment.simple.mirror,
            transpose_only=augment.simple.transpose) \
         if augment.simple is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='gaussian',
            var=augment.noise_gaussian.var,
            clip=False) \
         if augment.noise_gaussian is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='s&p',
            amount=augment.noise_saltpepper.amount,
            clip=False) \
         if augment.noise_saltpepper is not None else NoOp()) +

        (gp.IntensityAugment(
            raw,
            scale_min=augment.intensity.scale[0],
            scale_max=augment.intensity.scale[1],
            shift_min=augment.intensity.shift[0],
            shift_max=augment.intensity.shift[1],
            z_section_wise=False,
            clip=False) \
         if augment.intensity is not None else NoOp()) +

        AddParentVectors(
            tracks,
            parent_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            radius=config.train.parent_radius,
            # move_radius=20,
            move_radius=config.train.move_radius,
            dense=not config.general.sparse) +

        (gp.Reject(
            mask=cell_mask,
            min_masked=0.0001,
            # reject_probability=augment.reject_empty_prob
        ) \
         if config.general.sparse else NoOp()) +

        # gp.Reject(
        #     # already done by random_location ensure_nonempty arg
        #     # no, not done, elastic augment changes roi, point might
        #     # be in outer part that gets cropped
        #     ensure_nonempty=center_tracks,
        #     # reject_probability=augment.reject_empty_prob
        # ) +

        gp.RasterizePoints(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=config.train.rasterize_radius,
                mode='peak')) +

        gp.RasterizePoints(
            tracks,
            cell_center,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.1,) + voxel_size[1:],
                mode='point')) +

        (gp.PreCache(
            cache_size=config.train.cache_size,
            num_workers=config.train.job.num_workers) \
         if config.train.job.num_workers > 1 else NoOp())
    )

    if config.train.val_log_step is not None:
        val_pipeline = (
            tuple(val_sources) +
            gp.RandomProvider() +

            AddParentVectors(
                tracks,
                parent_vectors,
                cell_mask,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                radius=config.train.parent_radius,
                move_radius=config.train.move_radius,
                dense=not config.general.sparse) +

            (gp.Reject(
                mask=cell_mask,
                min_masked=0.0001,
                # reject_probability=augment.reject_empty_prob
            ) \
             if config.general.sparse else NoOp()) +

            gp.Reject(
                # already done by random_location ensure_nonempty arg
                # no, not done, elastic augment changes roi, point might
                # be in outer part that gets cropped
                ensure_nonempty=center_tracks,
                # reject_probability=augment.reject_empty_prob
            ) +

            gp.RasterizePoints(
                tracks,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=config.train.rasterize_radius,
                    mode='peak')) +

            gp.RasterizePoints(
                tracks,
                cell_center,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=(0.1,) + voxel_size[1:],
                    mode='point')) +

            (gp.PreCache(
                cache_size=config.train.cache_size,
                num_workers=1) \
             if config.train.job.num_workers > 1 else NoOp())
        )

    try:
        checkpoint_iteration = int(tf.train.latest_checkpoint(config.general.setup_dir).split("_checkpoint_")[-1])
    except:
        checkpoint_iteration = 0

    if config.train.val_log_step is not None:
        pipeline = (
            (train_pipeline, val_pipeline) +
            gp.TrainValProvider(step=config.train.val_log_step,
                                init_step=checkpoint_iteration))
    else:
        pipeline = train_pipeline


    inputs={
        net_names['anchor']: anchor,
        net_names['raw']: raw,
        net_names['gt_parent_vectors']: parent_vectors,
        net_names['gt_cell_indicator']: cell_indicator,
        net_names['gt_cell_center']: cell_center,
        net_names['cell_mask']: cell_mask,
    }
    outputs={
        net_names['parent_vectors']: pred_parent_vectors,
        net_names['cell_indicator']: pred_cell_indicator,
        net_names['raw_cropped']: raw_cropped,
        net_names['maxima']: maxima,
    }
    if config.train.cell_density:
        outputs[net_names['cell_density']] = pred_cell_density
        inputs[net_names['gt_cell_density']] = cell_density


    gradients={
        # net_names['parent_vectors']: grad_parent_vectors,
        # net_names['cell_indicator']: grad_cell_indicator,
    }

    snapshot_datasets = {
        raw: 'volumes/raw',
        raw_cropped: 'volumes/raw_cropped',
        parent_vectors: 'volumes/parent_vectors',
        cell_indicator: 'volumes/cell_indicator',
        cell_center: 'volumes/cell_center',
        cell_mask: 'volumes/cell_mask',
        pred_parent_vectors: 'volumes/pred_parent_vectors',
        pred_cell_indicator: 'volumes/pred_cell_indicator',
        maxima: 'volumes/maxima',
        # grad_parent_vectors: 'volumes/grad_parent_vectors',
        # grad_cell_indicator: 'volumes/grad_cell_indicator',
    }
    if config.train.cell_density:
        snapshot_datasets[cell_density] = 'volumes/cell_density'
        snapshot_datasets[pred_cell_density] = 'volumes/pred_cell_density'

    pipeline = (
        pipeline +
        gp.tensorflow.Train(
            os.path.join(config.general.setup_dir, 'train_net'),
            optimizer=net_names['optimizer'],
            loss=net_names['loss'],
            summary=summaries,
            inputs=inputs,
            outputs=outputs,
            gradients=gradients,
            log_dir=os.path.join(config.general.setup_dir, "train"),
            val_log_step=config.train.val_log_step,
            save_every=config.train.checkpoint_stride) +

        # visualize
        # gp.IntensityScaleShift(raw_cropped, scale=100.0, shift=0) +
        gp.Snapshot(snapshot_datasets,
            output_dir=os.path.join(config.general.setup_dir, 'snapshots'),
            output_filename='snapshot_{iteration}.hdf',
            additional_request=snapshot_request,
            every=config.train.snapshot_stride,
            dataset_dtypes={
                maxima: np.uint8
            }) +
        gp.PrintProfilingStats(every=config.train.profiling_stride)
    )

    with gp.build(pipeline):

        logger.info("Starting training...")
        for i in range(trained_until, config.train.max_iterations):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)


def get_sources(config, raw, anchor, tracks, center_tracks, data_sources,
                cell_density=None):
    sources = []
    for ds in data_sources:
        d = ds.datafile.filename
        voxel_size = gp.Coordinate(ds.voxel_size)
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        logger.info("loading data %s", d)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_zarr = os.path.join(
            d, data_config['general']['zarr_file'])
        filename_tracks = os.path.join(
            d, data_config['general']['tracks_file'])
        filename_daughters = os.path.join(
            d, data_config['general']['daughter_cells_file'])
        logger.info("creating source: %s (%s, %s, %s), divisions?: %s",
                    filename_zarr, ds.datafile.group,
                    filename_tracks, filename_daughters,
                    config.train.augment.divisions)
        file_size = data_config['general']['shape']
        limit_to_roi = gp.Roi(offset=ds.roi.offset, shape=ds.roi.shape)
        logger.info("limiting to roi: %s", limit_to_roi)
        datasets = {
            raw: ds.datafile.group,
            anchor: ds.datafile.group
        }
        array_specs = {
            raw: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size),
            anchor: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)
        }
        if config.train.cell_density is not None:
            datasets[cell_density] = 'volumes/cell_density'
            array_specs[cell_density] = gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)
        file_source = gp.ZarrSource(
            filename_zarr,
            datasets=datasets,
            nested="nested" in ds.datafile.group,
            array_specs=array_specs) + \
            gp.Crop(raw, limit_to_roi) + \
            gp.Pad(raw, None)


        file_resolution = gp.Coordinate(data_config['general']['resolution'])
        # file_resolution = gp.Coordinate([1, 11, 1, 1])
        scale = np.array(voxel_size)/np.array(file_resolution)
        logger.info("scaling tracks by %s", scale)
        track_source = (merge_sources(
            file_source,
            tracks,
            center_tracks,
            filename_tracks,
            filename_tracks,
            limit_to_roi,
            scale=scale,
            use_radius=config.train.use_radius) +
                          gp.RandomLocation(
                              ensure_nonempty=center_tracks,
                              p_nonempty=config.train.augment.reject_empty_prob)
                        )

        if config.train.augment.divisions:
            div_source = (merge_sources(
                file_source,
                tracks,
                center_tracks,
                filename_tracks,
                filename_daughters,
                limit_to_roi,
                scale=scale,
                use_radius=config.train.use_radius) +
                          gp.RandomLocation(
                              ensure_nonempty=center_tracks,
                              p_nonempty=config.train.augment.reject_empty_prob)
                          )


            track_source = (track_source, div_source) + \
                gp.RandomProvider(probabilities=[0.75, 0.25])
            # gp.Pad(raw, gp.Coordinate((0, 80, 55, 55))) + \

        # pad = np.array((input_size[0], 140, 100, 100))# * np.array(voxel_size)
        source = track_source
        # source = (file_source, track_source) + \
        #          gp.MergeProvider() + \
        #          gp.Crop(raw, limit_to_roi) + \
        #          gp.Pad(raw, None)
        #                 # gp.Coordinate(pad))

        if cell_density is not None:
            source = source + \
                gp.Crop(cell_density, limit_to_roi) + \
                gp.Pad(cell_density, None)
                       # gp.Coordinate(pad))
        # source = source + \
        #     gp.RandomLocation(
        #         ensure_nonempty=center_tracks,
        #         p_nonempty=config.train.augment.reject_empty_prob
        #     )

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
                NormalizeMinMax(raw, mn=mn, mx=mx, interpolatable=False)
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

        sources.append(source)

    return sources


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(sys.stderr)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    train_until(config)
