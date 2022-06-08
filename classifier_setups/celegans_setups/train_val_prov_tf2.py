from __future__ import print_function
import itertools
import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import glob
import os
import time
import warnings

from natsort import natsorted
import numpy as np
import pymongo
import tensorflow as tf
import gunpowder as gp

from linajea import (checkOrCreateDB,
                     load_config,
                     parse_tracks_file)
from linajea.gunpowder import (Clip,
                               GetLabels,
                               NoOp,
                               NormalizeMinMax)
from util import (parse_tracks_file_by_class)

from resnet_3d import resnet_50
# from perceiver import Perceiver

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("once", category=FutureWarning)


logger = logging.getLogger(__name__)


def preceiver_model(config):
    dropout_rate = 0.2
    latent_dim = 512  # Size of the latent array.
    data_dim = int(np.prod(config.model.input_shape))
    encoding_dim = 5
    projection_dim = 256  # Embedding size of each element in the data and latent arrays.
    num_heads = 8  # Number of Transformer heads.
    ffn_units = [
        projection_dim,
        projection_dim,
    ]  # Size of the Transformer Feedforward network.
    num_transformer_blocks = 4
    num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
    classifier_units = [
        projection_dim,
        config.model.num_classes
    ]  # Size of the Feedforward network of the final classifier.

    num_bands = 6
    # maximum frequency, hyperparameter depending on how fine the data is
    max_freq = 10.
    base = 2
    model = Perceiver(
        config.model.input_shape,
        config.train.batch_size,
        data_dim,
        latent_dim,
        encoding_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        max_freq,
        num_bands,
        base,
        num_iterations,
        classifier_units,
    )
    return model


def train_until(config, output_folder):
    logger.info("cuda visibile devices %s", os.environ["CUDA_VISIBLE_DEVICES"])
    checkpoint_basename = os.path.join(output_folder, 'model_checkpoint')
    checkpoints = glob.glob(checkpoint_basename + '*')
    if checkpoints:
        checkpoints = natsorted(checkpoints)
        trained_until = int(checkpoints[-1].split('-')[-1].split('.')[0]) * \
            config.train.checkpoint_stride
    else:
        trained_until = 0
    if trained_until >= config.train.max_iterations:
        return

    # model = preceiver_model(config)
    model = resnet_50()

    raw = gp.ArrayKey('RAW')
    pred_labels = gp.ArrayKey('PRED_LABELS')
    pred_probs = gp.ArrayKey('PRED_PROBS')
    gt_labels = gp.ArrayKey('GT_LABELS')

    voxel_size = gp.Coordinate(config.train_data.data_sources[0].voxel_size)
    input_shape = config.model.input_shape
    input_shape_world = gp.Coordinate(input_shape)*voxel_size

    request = gp.BatchRequest()
    # request.add(raw, input_shape_world)
    # request[gt_labels] = gp.ArraySpec(nonspatial=True)

    input_specs = {
        raw: gp.ArraySpec(roi=gp.Roi((0,)*len(input_shape_world),
                                     input_shape_world),
                          voxel_size=voxel_size,
                          interpolatable=True, dtype=np.float32),
        gt_labels: gp.ArraySpec(nonspatial=True,
                                interpolatable=False, dtype=np.float32),
        pred_labels: gp.ArraySpec(nonspatial=True),
        pred_probs: gp.ArraySpec(nonspatial=True)
    }

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape_world)
    snapshot_request[pred_labels] = gp.ArraySpec(nonspatial=True)
    snapshot_request[gt_labels] = gp.ArraySpec(nonspatial=True)
    snapshot_request[pred_probs] = gp.ArraySpec(nonspatial=True)
    logger.info("snapshot request %s", snapshot_request)

    train_sources = get_sources(config, raw, config.train_data.data_sources, val=False)
    if config.train.val_log_step is not None:
        val_sources = get_sources(config, raw,
                                  config.validate_data.data_sources, val=True)

    augment = config.train.augment

    train_pipeline = (
        train_sources +

        # elastically deform the batch
        (gp.ElasticAugment(
            augment.elastic.control_point_spacing,
            augment.elastic.jitter_sigma,
            [augment.elastic.rotation_min*np.pi/180.0,
             augment.elastic.rotation_max*np.pi/180.0],
            rotation_3d=augment.elastic.rotation_3d,
            subsample=augment.elastic.subsample,
            spatial_dims=3) \
         if augment.elastic is not None else NoOp())  +

        # apply transpose and mirror augmentations
        (gp.SimpleAugment(
            mirror_only=augment.simple.mirror,
            transpose_only=augment.simple.transpose) \
         if augment.simple is not None else NoOp()) +

        (gp.ZoomAugment(
            factor_min=augment.zoom.factor_min,
            factor_max=augment.zoom.factor_max,
            spatial_dims=augment.zoom.spatial_dims,
            order={raw: 1,
                   }) \
         if augment.zoom is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='gaussian',
            var=augment.noise_gaussian.var,
            clip=False) \
         if augment.noise_gaussian is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='speckle',
            var=augment.noise_speckle.var,
            clip=False) \
         if augment.noise_speckle is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='s&p',
            amount=augment.noise_saltpepper.amount,
            clip=False) \
         if augment.noise_saltpepper is not None else NoOp()) +

        (gp.HistogramAugment(
            raw,
            # raw_tmp,
            range_low=augment.histogram.range_low,
            range_high=augment.histogram.range_high,
            z_section_wise=False) \
         if (augment.histogram is not None and not augment.histogram.after_int_aug) else NoOp())  +

        # # scale and shift the intensity of the raw array
        (gp.IntensityAugment(
            raw,
            scale_min=augment.intensity.scale[0],
            scale_max=augment.intensity.scale[1],
            shift_min=augment.intensity.shift[0],
            shift_max=augment.intensity.shift[1],
            z_section_wise=False,
            clip=False) \
         if augment.intensity is not None else NoOp())  +

        (gp.HistogramAugment(
            raw,
            # raw_tmp,
            range_low=augment.histogram.range_low,
            range_high=augment.histogram.range_high,
            z_section_wise=False) \
         if (augment.histogram is not None and augment.histogram.after_int_aug) else NoOp())  +


        GetLabels(raw, gt_labels) +

        gp.PreCache(
            cache_size=config.train.cache_size * config.train.batch_size,
            num_workers=config.train.job.num_workers) +

        gp.Stack(config.train.batch_size)
    )

    if config.train.val_log_step is not None:
        val_pipeline = (
            val_sources +

            GetLabels(raw, gt_labels) +

            gp.PreCache(
                cache_size=2 * config.train.batch_size,
                num_workers=1) +

            gp.Stack(config.train.batch_size)
        )

    opt = getattr(tf.keras.optimizers, config.optimizerTF2.optimizer)(
        **config.optimizerTF2.get_kwargs())
    logger.info("optimizer: %s", opt)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    try:
        checkpoint_iteration = int(tf.train.latest_checkpoint(os.path.dirname(os.path.join(output_folder, config.model.net_name))).split("_checkpoint_")[-1])
    except:
        checkpoint_iteration = 0

    if config.train.val_log_step is not None:
        pipeline = ((train_pipeline, val_pipeline) +
                    gp.TrainValProvider(step=config.train.val_log_step,
                                        init_step=checkpoint_iteration)
                    )
    else:
        pipeline = train_pipeline

    pipeline = (
        pipeline +

        gp.tensorflow2.Train(
            model=model,
            optimizer=opt,
            # summary=summaries,
            log_dir=output_folder,
            loss_object=loss,
            inputs={
                'raw': raw,
            },
            outputs={
                0: pred_labels,
                1: pred_probs,
            },
            loss_inputs={
                'gt_labels': gt_labels,

            },
            gradients={
                # 'pred_labels': pred_labels_gradients,
            },
            array_specs=input_specs,
            # array_specs={
            #     pred_labels: gp.ArraySpec(nonspatial=True),
            #     pred_probs: gp.ArraySpec(nonspatial=True)
            # },
            # auto_mixed_precision=config.train.use_auto_mixed_precision,
            checkpoint_basename=checkpoint_basename,
            save_every=config.train.checkpoint_stride,
            # val_log_step=config.train.val_log_step,
            # snapshot_every=config.train.snapshot_stride
        ) +


        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot({
            raw: 'volumes/raw',
            gt_labels: 'volumes/gt_labels',
            pred_labels: 'volumes/pred_labels',
            pred_probs: 'volumes/pred_probs'
            },
            output_dir=os.path.join(output_folder, 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=config.train.snapshot_stride,
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=config.train.profiling_stride)
    )

    logger.info("Starting training...")
    with gp.build(pipeline):
        logger.info("pipeline %s", pipeline)
        for i in range(trained_until, config.train.max_iterations):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)
    logger.info("Training finished")


def get_sources(config, raw, data_sources, val=False):

    if config.train_data.use_database:
        pass
        client = pymongo.MongoClient(host=config.general.db_host)

        locationsByClass = []
        labelsByClass = []
        for didx, data_source in enumerate(data_sources):
            db_meta_info = config.train_data.db_meta_info
            if data_source.db_name is None:
                db_name = checkOrCreateDB(
                    config.general.db_host,
                    db_meta_info.setup_dir,
                    data_source.datafile.filename,
                    db_meta_info.checkpoint,
                    db_meta_info.cell_score_threshold,
                    create_if_not_found=False)

                assert db_name is not None, "db for %s (%s) not found" % (
                    data_source, db_meta_info)
            else:
                db_name = data_source.db_name
            if "polar" in db_name and not config.model.with_polar:
                continue
            logger.info("loading %s from db %s (val %s)",
                        data_source.datafile.filename,
                        db_name, val)
            db = client[db_name]
            cells_db = db['nodes']
            cells = list(cells_db.find().sort('id', pymongo.ASCENDING))
            locationsByClass_t = {}
            labelsByClass_t = {}

            d = data_source.datafile.filename
            if not os.path.isdir(d):
                logger.info("trimming path %s", d)
                d = os.path.dirname(d)
            data_config = load_config(os.path.join(d, "data_config.toml"))
            file_resolution = gp.Coordinate(data_config['general']['resolution'])
            scale = np.array(data_source.voxel_size)/np.array(file_resolution)
            logger.info("scaling tracks by %s", scale)
            limit_to_roi = gp.Roi(offset=data_source.roi.offset,
                                  shape=data_source.roi.shape)
            logger.info("limiting to roi {}".format(limit_to_roi))

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
                    if "polar" in db_name:
                        locationsByClass_t.setdefault("polar", []).append(cell_pos)
                        labelsByClass_t.setdefault("polar", []).append(
                            config.model.num_classes)
                    else:
                        logger.warning(
                            "node candidate {} has no probable_gt_state, skipping".format(
                                cell_pos))
                        skipped += 1
            logger.info("{}/{} node candidates skipped (db: {}, sample: {})".format(
                skipped, len(cells), db_name,
                os.path.basename(data_source.datafile.filename)))
            locationsByClass.append(locationsByClass_t)
            labelsByClass.append(labelsByClass_t)
    else:
        locationsByClass = []
        labelsByClass = []
        for didx, data_source in enumerate(data_sources):
            d = data_source.datafile.filename
            logger.info("loading data %s (val: %s)", d, val)
            if not os.path.isdir(d):
                logger.info("trimming path %s", d)
                d = os.path.dirname(d)
            data_config = load_config(os.path.join(d, "data_config.toml"))
            filename_tracks = os.path.join(
                d, data_config['general']['tracks_file'])
            if "polar" in data_source.datafile.filename:
                if not config.model.with_polar:
                    continue
                filename_tracks = os.path.splitext(filename_tracks)[0] + "_polar.txt"
            else:
                filename_tracks = os.path.splitext(filename_tracks)[0] + "_div_state.txt"
            logger.info("loading from file %s", filename_tracks)
            limit_to_roi = gp.Roi(offset=data_source.roi.offset,
                                  shape=data_source.roi.shape)
            logger.info("limiting to roi {}".format(limit_to_roi))
            file_resolution = gp.Coordinate(data_config['general']['resolution'])
            scale = np.array(data_source.voxel_size)/np.array(file_resolution)
            logger.info("scaling tracks by %s", scale)
            if "polar" in data_source.datafile.filename:
                locations, track_info = parse_tracks_file(
                    filename_tracks, scale=scale,
                    limit_to_roi=limit_to_roi)

                locationsByClass_t = {}
                labelsByClass_t = {}
                for idx, cell in enumerate(track_info):
                    locationsByClass_t.setdefault("polar", []).append(locations[idx])
                    labelsByClass_t.setdefault("polar", []).append(config.model.num_classes)
                logger.info("%s: #class polar: %s",
                            os.path.basename(data_source.datafile.filename),
                            len(locationsByClass_t["polar"]))
            else:
                _, _, locationsByClass_t, labelsByClass_t = \
                    parse_tracks_file_by_class(
                        filename_tracks, num_classes=config.model.num_classes,
                        scale=scale, limit_to_roi=limit_to_roi)
                for cls in range(config.model.num_classes):
                    logger.info("%s: #class %s: %s",
                                os.path.basename(data_source.datafile.filename),
                                config.model.classes[cls],
                                len(locationsByClass_t[cls]))
            locationsByClass.append(locationsByClass_t)
            labelsByClass.append(labelsByClass_t)

    augment = config.train.augment
    sources = []
    probs = []
    for didx, data_source in enumerate(data_sources):
        d = data_source.datafile.filename
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_zarr = os.path.join(d, data_config['general']['zarr_file'])
        mn = data_config['stats'].get(augment.min_key, augment.norm_min)
        mx = data_config['stats'].get(augment.max_key, augment.norm_max)

        source_cls = []
        if "polar" in data_source.datafile.filename:
            if not config.model.with_polar:
                continue
            rnge = ["polar"]
        else:
            rnge = range(config.model.num_classes)
        print(rnge, data_source.datafile.filename)
        for cls in rnge:
            source = (
                gp.ZarrSource(
                    filename_zarr,
                    datasets = {
                        raw: data_source.datafile.group,
                    },
                    array_specs={
                        raw: gp.ArraySpec(interpolatable=True,
                                          voxel_size=data_source.voxel_size,
                                          )
                    },
                    nested="nested" in data_source.datafile.group,
                    load_to_mem=False,
                ) +
                Clip(raw, mn=mn, mx=mx) +
                NormalizeMinMax(raw, mn=mn, mx=mx) +
                gp.Pad(raw, gp.Coordinate(config.model.pad_raw)) +
                gp.SpecifiedLocation(
                    locationsByClass[didx][cls], choose_randomly=True,
                    extra_data=labelsByClass[didx][cls],
                    jitter=augment.jitter.jitter if augment.jitter is not None else None)
                )
            source_cls.append(source)
        if "polar" in data_source.datafile.filename:
            source_cls = tuple(source_cls) + \
                gp.RandomProvider()
            probs.append(config.model.class_sampling_weights[-1])
        else:
            source_cls = tuple(source_cls) + \
                gp.RandomProvider(probabilities=config.model.class_sampling_weights[:-1])
            probs.append(sum(config.model.class_sampling_weights[:-1]))
        sources.append(source_cls)

    sources = (
        tuple(sources) +
        gp.RandomProvider(probabilities=probs)
    )
    return sources


def get_sources_dataset(config, raw, data_sources, val=False):

    locations = []
    labels = []
    for didx, data_source in enumerate(data_sources):
        d = data_source.datafile.filename
        logger.info("loading data %s (val: %s)", d, val)
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_tracks = os.path.join(
            d, data_config['general']['tracks_file'])
        if "polar" in data_source.datafile.filename:
            continue
        else:
            filename_tracks = os.path.splitext(filename_tracks)[0] + "_div_state.txt"
        logger.info("loading from file %s", filename_tracks)
        limit_to_roi = gp.Roi(offset=data_source.roi.offset,
                              shape=data_source.roi.shape)
        logger.info("limiting to roi {}".format(limit_to_roi))
        file_resolution = gp.Coordinate(data_config['general']['resolution'])
        scale = np.array(data_source.voxel_size)/np.array(file_resolution)
        logger.info("scaling tracks by %s", scale)
        _, _, locationsByClass_t, _ = \
            parse_tracks_file_by_class(
                filename_tracks, num_classes=config.model.num_classes,
                scale=scale, limit_to_roi=limit_to_roi)
        for cls in range(config.model.num_classes):
            logger.info("%s: #class %s: %s",
                        os.path.basename(data_source.datafile.filename),
                        config.model.classes[cls],
                        len(locationsByClass_t[cls]))
        locations_t = list(itertools.chain(*locationsByClass_t))
        locations_t2 = []
        for loc in locations_t:
            if not val and loc[0] % 10 < 5:
                locations_t2.append(loc)
            elif val and loc[0] % 10 >= 5:
                locations_t2.append(loc)
        labels_t = [didx]*len(locations_t2)

        locations.append(locations_t2)
        labels.append(labels_t)

    augment = config.train.augment
    sources = []
    for didx, data_source in enumerate(data_sources):
        d = data_source.datafile.filename
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_zarr = os.path.join(d, data_config['general']['zarr_file'])
        mn = data_config['stats'].get(augment.min_key, augment.norm_min)
        mx = data_config['stats'].get(augment.max_key, augment.norm_max)

        if "polar" in data_source.datafile.filename:
            continue
        source = (
            gp.ZarrSource(
                filename_zarr,
                datasets = {
                    raw: data_source.datafile.group,
                },
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True,
                                      voxel_size=data_source.voxel_size,
                                      )
                },
                nested="nested" in data_source.datafile.group,
                load_to_mem=False,
            ) +
            Clip(raw, mn=mn, mx=mx) +
            NormalizeMinMax(raw, mn=mn, mx=mx) +
            gp.Pad(raw, gp.Coordinate(config.model.pad_raw)) +
            gp.SpecifiedLocation(
                locations[didx], choose_randomly=True,
                extra_data=labels[didx],
                jitter=augment.jitter.jitter if augment.jitter is not None else None)
        )
        sources.append(source)

    sources = (
        tuple(sources) +
        gp.RandomProvider()
    )
    return sources
