from __future__ import print_function
import argparse
import json
from linajea.gunpowder import (TracksSource, AddParentVectors,
                               ShiftAugment, RandomLocationExcludeTime)
import gunpowder as gp
import logging
import math
import os
import tensorflow as tf

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def merge_sources(
        data_dir,
        source,
        tracks,
        center_tracks,
        track_file,
        ):
    return (
        # raw
        (source,
         # tracks
         TracksSource(
             os.path.join(
                 data_dir,
                 'tracks',
                 track_file),
             tracks),
         # center tracks
         TracksSource(
             os.path.join(
                 data_dir,
                 'tracks',
                 track_file),
             center_tracks)
         ) +
        gp.MergeProvider() +
        gp.Pad(tracks, None) +
        gp.Pad(center_tracks, None))


def train_until(
        max_iteration,
        data_dir,
        setup_dir,
        zarr_file,
        tracks_file,
        daughter_cells_file,
        exclude_times,
        shift,
        divisions,
        snapshot_frequency=1000):

    # Get the latest checkpoint.
    if tf.train.latest_checkpoint(setup_dir):
        trained_until = int(tf.train.latest_checkpoint(
            setup_dir).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open(setup_dir + '/train_net_config.json', 'r') as f:
        net_config = json.load(f)

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

    voxel_size = gp.Coordinate((1, 5, 1, 1))
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
    print("Center size: {}".format(center_size))
    print("Output size 1: {}".format(output_size_1))

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(tracks, output_size_1)
    request.add(center_tracks, center_size)
    request.add(parent_vectors, output_size_1)
    request.add(cell_indicator, output_size_1)
    request.add(cell_mask, output_size_1)
    print("REQUEST: %s" % str(request))
    snapshot_request = gp.BatchRequest({
        raw: request[raw],
        pred_parent_vectors: request[parent_vectors],
        pred_cell_indicator: request[parent_vectors],
        grad_parent_vectors: request[parent_vectors],
        grad_cell_indicator: request[parent_vectors]
    })
    snapshot_request.add(maxima, output_size_2)
    print("Snapshot request: %s" % str(snapshot_request))
    source = (gp.ZarrSource(
            os.path.join(
                data_dir,
                zarr_file),
            {raw: 'raw'},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)}) +
              gp.Normalize(raw))
    sources = (merge_sources(
            data_dir,
            source,
            tracks,
            center_tracks,
            tracks_file) +
        RandomLocationExcludeTime(
            raw=raw,
            time_interval=exclude_times,
            ensure_nonempty=center_tracks))
    if divisions:
        div_sources = (merge_sources(
                data_dir,
                source,
                tracks,
                center_tracks,
                daughter_cells_file) +
            RandomLocationExcludeTime(
                raw=raw,
                time_interval=exclude_times,
                ensure_nonempty=center_tracks))

        sources = (
            (sources, div_sources) +
            gp.RandomProvider(probabilities=[0.75, 0.25])
            )
    pipeline = (
        sources +

        # augment
        gp.ElasticAugment(
            control_point_spacing=(5, 10, 10),
            jitter_sigma=(1, 1, 1),
            rotation_interval=[0, math.pi/2.0],
            subsample=8)
        )
    if shift:
        pipeline = pipeline + ShiftAugment(
            prob_slip=0.3,
            prob_shift=0.3,
            sigma=[0, 5, 5, 5],
            shift_axis=0)
    pipeline = (
        pipeline +
        gp.SimpleAugment(mirror_only=[1, 2, 3], transpose_only=[2, 3]) +
        gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +

        AddParentVectors(
            tracks,
            parent_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            radius=(0.1, 10, 10, 10),
            move_radius=10) +

        gp.RasterizePoints(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.1, 5, 5, 5),
                mode='peak')) +

        gp.Reject(
                ensure_nonempty=tracks,
                mask=cell_mask,
                min_masked=0.0001) +

        # train
        gp.PreCache(
            cache_size=40,
            num_workers=10) +
        gp.tensorflow.Train(
            setup_dir + '/train_net',
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
            save_every=25000) +

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
            output_dir=setup_dir + '/snapshots',
            output_filename='snapshot_{iteration}.hdf',
            additional_request=snapshot_request,
            every=snapshot_frequency) +
        gp.PrintProfilingStats(every=10)
    )

    with gp.build(pipeline):

        print("Starting training...")
        for i in range(max_iteration - trained_until):
            pipeline.request_batch(request)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--iterations', type=int,
                        help='number ofiterations to train')
    parser.add_argument('--snap_frequency', type=int,
                        help='number of iterations to save snapshots',
                        default=1000)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train_until(
        args.iterations,
        config['data_dir'],
        config['setup_dir'],
        config['zarr_file'],
        config['tracks_file'],
        config['daughter_cells_file'],
        config['exclude_times'],
        config['shift'],
        config['divisions'],
        snapshot_frequency=args.snap_frequency)
