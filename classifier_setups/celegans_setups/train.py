from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import json
import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import math
import os
import sys
import time

import numpy as np
import pymongo
import tensorflow as tf
import gunpowder as gp

from linajea import (checkOrCreateDB,
                     load_config,
                     parse_limit_roi)
from linajea.gunpowder import (Clip,
                               GetLabels,
                               NoOp,
                               NormalizeMinMax)
from util import (parse_tracks_file_by_class)

logger = logging.getLogger(__name__)


def train_until(config, output_folder):
    logger.info("cuda visibile devices %s", os.environ["CUDA_VISIBLE_DEVICES"])
    if tf.train.latest_checkpoint(output_folder):
        trained_until = int(
            tf.train.latest_checkpoint(output_folder).split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= config['model']['max_iterations']:
        return

    net_config = load_config(os.path.join(
        output_folder, config['model']['net_name'] + '_config.json'))
    net_names = load_config(os.path.join(
        output_folder, config['model']['net_name'] + '_names.json'))
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
                elif "network" in k:
                    summaries[k] = (v, 25)
                elif "histo" in k:
                    summaries[k] = (v, 50)
                elif "image" in k:
                    summaries[k] = (v, 50)
                else:
                    summaries[k] = (v, 10)

    raw = gp.ArrayKey('RAW')
    pred_labels = gp.ArrayKey('PRED_LABELS')
    pred_probs = gp.ArrayKey('PRED_PROBS')
    gt_labels = gp.ArrayKey('GT_LABELS')

    voxel_size = gp.Coordinate(config['data']['voxel_size'])
    input_shape_world = gp.Coordinate(net_config['input_shape'])*voxel_size
    input_shape = gp.Coordinate(net_config['input_shape'])

    request = gp.BatchRequest()

    input_specs = {
        raw: gp.ArraySpec(roi=gp.Roi((0,)*len(input_shape_world),
                                     input_shape_world),
                          voxel_size=voxel_size,
                          interpolatable=True, dtype=np.float32),
        gt_labels: gp.ArraySpec(nonspatial=True,
                                interpolatable=False, dtype=np.int32),
        pred_labels: gp.ArraySpec(nonspatial=True),
        pred_probs: gp.ArraySpec(nonspatial=True)
    }

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape_world)
    snapshot_request[pred_labels] = gp.ArraySpec(nonspatial=True)
    snapshot_request[gt_labels] = gp.ArraySpec(nonspatial=True)
    snapshot_request[pred_probs] = gp.ArraySpec(nonspatial=True)
    logger.info("snapshot request %s", snapshot_request)

    if config['data']['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif config['data']['input_format'] == "zarr":
        sourceNode = gp.ZarrSource
    else:
        raise RuntimeError("Invalid input format %s", config['data']['input_format'])

    limit_to_roi = parse_limit_roi(config['training'].get('roi', {}))
    if limit_to_roi is not None:
        logger.info("limiting to roi {}".format(limit_to_roi))
    if config['training']['use_database']:
        client = pymongo.MongoClient(host=config['general']['db_host'])

        locationsByClass = []
        labelsByClass = []
        for didx, data_dir in enumerate(config['data']['train_data_dirs']):
            linajea_config = config['training']['linajea_config']
            linajea_config['general']['db_host'] = config['general' ]['db_host']
            db_name = checkOrCreateDB(linajea_config,
                                      data_dir,
                                      create_if_not_found=False)
            assert db_name is not None, "db for %s not found" % data_dir
            logger.info("loading from db %s", db_name)
            db = client[db_name]
            cells_db = db['nodes']
            cells = list(cells_db.find().sort('id', pymongo.ASCENDING))
            locationsByClass_t = {}
            labelsByClass_t = {}

            if os.path.isdir(data_dir):
                data_config = load_config(
                    os.path.join(data_dir, "data_config.toml"))
            else:
                data_config = load_config(
                    os.path.join(os.path.dirname(data_dir), "data_config.toml"))
            file_resolution = gp.Coordinate(data_config['general']['resolution'])
            scale = np.array(voxel_size)/np.array(file_resolution)
            logger.info("scaling tracks by %s", scale)

            skipped = 0
            for cell in cells:
                cell_pos = np.array((cell['t'], cell['z'],
                                     cell['y'], cell['x'])) * scale
                if limit_to_roi is not None and \
                   not limit_to_roi.contains(cell_pos):
                    logger.debug("cell %s outside of roi %s, skipping",
                                 cell_pos, limit_to_roi)
                    continue
                try:
                    cell_cls = cell['probable_gt_state']
                    locationsByClass_t.setdefault(cell_cls, []).append(cell_pos)
                    labelsByClass_t.setdefault(cell_cls, []).append(cell_cls)
                except:
                    logger.warning(
                        "node candidate {} has no probable_gt_state, skipping".format(
                            cell_pos))
                    skipped += 1
            logger.info("{}/{} node candidates skipped (db: {}, sample: {})".format(
                skipped, len(cells), db_name, os.path.basename(data_dir)))
    else:
        locationsByClass = []
        labelsByClass = []
        for didx, data_dir in enumerate(config['data']['train_data_dirs']):
            if os.path.isdir(data_dir):
                data_config = load_config(
                    os.path.join(data_dir, "data_config.toml"))
                filename_tracks = os.path.join(
                    data_dir, data_config['general']['tracks_file'])
            else:
                data_config = load_config(
                    os.path.join(os.path.dirname(data_dir), "data_config.toml"))
                filename_tracks = os.path.join(
                    os.path.dirname(data_dir), data_config['general']['tracks_file'])
            filename_tracks = os.path.splitext(filename_tracks)[0] + "_div_state.txt"

            file_resolution = gp.Coordinate(data_config['general']['resolution'])
            scale = np.array(voxel_size)/np.array(file_resolution)
            logger.info("scaling tracks by %s", scale)

            logger.info("loading from file %s", filename_tracks)
            fields = "header"
            _, _, locationsByClass_t, labelsByClass_t = \
                parse_tracks_file_by_class(
                    filename_tracks, csv_fields=fields,
                    num_classes=config['model']['num_classes'],
                    scale=scale,
                    limit_to_roi=limit_to_roi)
            locationsByClass.append(locationsByClass_t)
            labelsByClass.append(labelsByClass_t)
            for cls in range(config['model']['num_classes']):
                logger.info("%s: #class %s: %s",
                            os.path.basename(data_dir),
                            config['data']['classes'][cls],
                            len(locationsByClass[didx][cls]))


    augmentation = config['training'].get('augmentation', {})

    sources = []
    # config['data]['train_data_dirs'] = [config['data]['train_data_dirs'][0]]
    for didx, data_dir in enumerate(config['data']['train_data_dirs']):
        if os.path.isdir(data_dir):
            data_config = load_config(
                os.path.join(data_dir, "data_config.toml"))
            filename_zarr = os.path.join(
                data_dir, data_config['general']['zarr_file'])
        else:
            data_config = load_config(
                os.path.join(os.path.dirname(data_dir), "data_config.toml"))
            filename_zarr = os.path.join(
                os.path.dirname(data_dir), data_config['general']['zarr_file'])

        mn = data_config['stats'].get(
            config['data'].get('min_key'),
            config['data' ]['norm_min'])
        mx = data_config['stats'].get(
            config['data'].get('max_key'),
            config['data']['norm_max'])

        source = tuple(
            sourceNode(
                filename_zarr,
                datasets = {
                    raw: config['data']['raw_key'],
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True,
                                      voxel_size=voxel_size,
                                      )
                },
                nested="nested" in config['data']['raw_key'],
                load_to_mem=False,
            ) +
            Clip(raw, mn=mn, mx=mx) +
            NormalizeMinMax(raw, mn=mn, mx=mx) +
            gp.Pad(raw, gp.Coordinate(config['model']['pad_raw'])) +
            gp.SpecifiedLocation(
                locationsByClass[didx][cls], choose_randomly=True,
                extra_data=labelsByClass[didx][cls],
                # jitter=augmentation['jitter'].get('jitter'))
                jitter=augmentation.get('jitter', {}).get('jitter'))
            for cls in range(config['model']['num_classes'])) + \
            gp.RandomProvider()
        sources.append(source)

    optimizer_args = None
    if config['training']['auto_mixed_precision']:
        optimizer_args = (config['optimizer']['optimizer'],
                          {
                              'args': config['optimizer']['args'],
                              'kwargs': config['optimizer']['kwargs']
                          })

    pipeline = (
        tuple(sources) +
        gp.RandomProvider() +
        # gp.RandomProvider(probabilities=[1, 1, 1]) +

        # elastically deform the batch
        (gp.ElasticAugment(
            augmentation['elastic']['control_point_spacing'],
            augmentation['elastic']['jitter_sigma'],
            [augmentation['elastic']['rotation_min']*np.pi/180.0,
             augmentation['elastic']['rotation_max']*np.pi/180.0],
            subsample=augmentation['elastic']['subsample']) \
        if augmentation.get('elastic') is not None else NoOp())  +

        # apply transpose and mirror augmentations
        (gp.SimpleAugment(
            mirror_only=augmentation['simple'].get("mirror"),
            transpose_only=augmentation['simple'].get("transpose")) \
         if augmentation.get('simple') is not None else NoOp()) +

        # # scale and shift the intensity of the raw array
        (gp.IntensityAugment(
            raw,
            scale_min=augmentation['intensity']['scale'][0],
            scale_max=augmentation['intensity']['scale'][1],
            shift_min=augmentation['intensity']['shift'][0],
            shift_max=augmentation['intensity']['shift'][1],
            z_section_wise=False) \
        if augmentation.get('intensity') is not None else NoOp())  +

        GetLabels(raw, gt_labels) +

        gp.PreCache(
            cache_size=config['model']['cache_size'],
            num_workers=config['model']['num_workers']) +

        # gp.Stack(config['model']['batch_size']) +
        (gp.tensorflow.TFData(batch_size=config['model']['batch_size']) \
         if config['training'].get('use_tf_data') else
         gp.Stack(config['model']['batch_size'])) +

        gp.tensorflow.Train(
            os.path.join(output_folder, config['model']['net_name']),
            optimizer=net_names['optimizer'],
            summary=summaries,
            log_dir=output_folder,
            loss=net_names['loss'],
            is_training=net_names['is_training'],
            inputs={
                net_names['raw']: raw,
                net_names['gt_labels']: gt_labels,
            },
            outputs={
                net_names['pred_labels']: pred_labels,
                net_names['pred_probs']: pred_probs,
            },
            gradients={
                # net_names['pred_labels']: pred_labels_gradients,
            },
            array_specs=input_specs,
            auto_mixed_precision=config['training']['auto_mixed_precision'],
            optimizer_args=optimizer_args,
            use_tf_data=config['training']['use_tf_data'],
            save_every=config['model']['checkpoints'],
            snapshot_every=config['model']['snapshots']) +


        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot({
            raw: 'volumes/raw',
            gt_labels: 'volumes/gt_labels',
            pred_labels: 'volumes/pred_labels',
            pred_probs: 'volumes/pred_probs'
            },
            output_dir=os.path.join(output_folder, 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=config['model']['snapshots'],
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=config['model']['profiling'])
    )

    logger.info("Starting training...")
    with gp.build(pipeline):
        logger.info("pipeline %s", pipeline)
        for i in range(trained_until, config['model']['max_iterations']):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)
    logger.info("Training finished")
