import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import os
import random
import time
import warnings

import numpy as np
import pymongo
import sklearn.utils
import tensorflow as tf
import zarr

import gunpowder as gp
from linajea import (load_config,
                     parse_tracks_file)
from util import parse_tracks_file_by_class

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("once", category=FutureWarning)


logger = logging.getLogger(__name__)


def cell_division_input_fn(config, sample, prefix, node_collection):

    def get_sample(it):
        if config.inference.use_database:
            if it >= len(cell_list) or \
               (config.predict.max_samples and
                it >= config.predict.max_samples):
                logger.info("done eval/predict database: #cells: %s", len(cell_list))
                client.close()
                return -1, -1
            cell = cell_list[it]
            if cell['score'] < config.inference.db_meta_info.cell_score_threshold and \
               'unedited' not in config.inference.data_source.db_name:
                return None, None
            # skip if prediction already exists
            if prefix + config.model.classes[0] in cell:
                return None, None

            sample = np.array([cell['t'],
                               cell['z'], cell['y'], cell['x']],
                              dtype=np.int32)
            if limit_to_roi is not None and \
               not limit_to_roi.contains(sample):
                logger.info("cell %s outside of roi %s, skipping",
                            sample, limit_to_roi)
                return None, None
            # if not cell.get('in_mask', True):
            #     return None, None

            label = -1
        else:
            if "polar" in config.inference.data_source.datafile.filename:
                sample_id = it
                if sample_id >= len(locationsByClass["polar"]):
                    raise StopIteration
                label = labelsByClass["polar"][sample_id]
                sample = locationsByClass["polar"][sample_id].astype(np.int32)
            elif it < len(locationsByClass[1]):
                sample_id = it
                label = labelsByClass[1][sample_id]
                sample = locationsByClass[1][sample_id].astype(np.int32)
                # logger.info("1, %s", label)
            elif it < len(locationsByClass[1]) + len(locationsByClass[2]):
                sample_id = it - len(locationsByClass[1])
                label = labelsByClass[2][sample_id]
                sample = locationsByClass[2][sample_id].astype(np.int32)
                # logger.info("2, %s", label)
            else:
                sample_id = \
                    it - len(locationsByClass[1]) - len(locationsByClass[2])
                if sample_id >= len(locationsByClass[0]):
                    raise StopIteration
                label = labelsByClass[0][sample_id]
                sample = locationsByClass[0][sample_id].astype(np.int32)
                # logger.info("0, %s", label)
            if limit_to_roi is not None and \
               not limit_to_roi.contains(sample):
                logger.info("cell %s outside of roi %s, skipping",
                            sample, limit_to_roi)
                return None, None

            if config.predict.max_samples and \
               it >= config.predict.max_samples:
                return None, None

        logger.debug("sample location (world): %s, class: %d", sample, label)

        return sample, label


    def generator_fn():
        it = 0
        while True:
            try:
                sampleT, label = get_sample(it)
            except StopIteration as e:
                logger.info("%s", e)
                return
            if isinstance(sampleT, int) and sampleT == -1 and label == -1:
                return
            it += 1
            if sampleT is None and label is None:
                continue
            sample = np.round(sampleT/voxel_size).astype(np.uint32)
            # sample = np.round(sampleT).astype(np.uint32)
            pad = False
            for i in range(len(input_shape)):
                if sample[i] - input_shape[i]//2 < 0 or \
                   sample[i] + input_shape[i]//2 >= data_shape[i]:
                    # sample to close to border of volume
                    # ignore for training/eval, pad for prediction
                    logger.debug(
                        "sample outside data ({}, {}, {}, {}, {})".format(
                            i, sample[i],
                            sample[i] - input_shape[i]//2,
                            sample[i] + input_shape[i]//2,
                            data_shape[i]))
                    pad = True

            patch = data[
                max(0, sample[0]-input_shape[0]//2):
                min(data_shape[0], int(np.ceil(sample[0]+input_shape[0]/2.0))),
                max(0, sample[1]-input_shape[1]//2):
                min(data_shape[1], int(np.ceil(sample[1]+input_shape[1]/2.0))),
                max(0,sample[2]-input_shape[2]//2):
                min(data_shape[2], int(np.ceil(sample[2]+input_shape[2]/2.0))),
                max(0, sample[3]-input_shape[3]//2):
                min(data_shape[3], int(np.ceil(sample[3]+input_shape[3]/2.0)))
            ]

            patch = patch.astype(np.float32)
            logger.warning("make sure same normalization as in train is used!")
            augment = config.train.augment
            mn = data_config['stats'].get(augment.min_key, augment.norm_min)
            mx = data_config['stats'].get(augment.max_key, augment.norm_max)
            patch1 = np.clip(patch, mn, mx)
            patch = ((patch1 - mn) / (mx - mn))

            if pad:
                padding = []
                for i in range(len(input_shape)):
                    before = 0
                    after = 0
                    if sample[i] - input_shape[i]//2 < 0:
                        before = abs(sample[i] - input_shape[i]//2)
                    if sample[i] + input_shape[i]//2 >= data_shape[i]:
                        after = abs(
                            sample[i] + int(np.ceil(input_shape[i]/2)) - data_shape[i])
                    padding.append((before, after))
                logger.info("padding {} {} {}".format(padding, patch.shape,
                                                       input_shape))
                patch = np.pad(patch, padding, 'constant')

            if config.predict.test_time_reps > 1:
                if np.random.rand() > 0.5:
                    patch = np.flip(patch, axis=2)
                if np.random.rand() > 0.5:
                    patch = np.flip(patch, axis=3)

                int_aug = config.train.augment.intensity
                scale = np.random.uniform(low=int_aug.scale[0], high=int_aug.scale[1])
                shift = np.random.uniform(low=int_aug.shift[0], high=int_aug.shift[1])
                patch = patch.mean() + (patch-patch.mean())*scale + shift

            yield {"position": sampleT, "input": patch}, label

    input_shape = config.model.input_shape
    is_polar = "polar" in sample
    if is_polar:
        sample = sample.replace("_polar", "")
    if os.path.isdir(sample):
        data_config = load_config(
            os.path.join(sample, "data_config.toml"))
        filename_zarr = os.path.join(
            sample, data_config['general']['zarr_file'])
        filename_tracks = os.path.join(
            sample, data_config['general']['tracks_file'])
    else:
        data_config = load_config(
            os.path.join(os.path.dirname(sample), "data_config.toml"))
        filename_zarr = os.path.join(
            os.path.dirname(sample), data_config['general']['zarr_file'])
        filename_tracks = os.path.join(
            os.path.dirname(sample), data_config['general']['tracks_file'])

    limit_to_roi = gp.Roi(offset=config.inference.data_source.roi.offset,
                          shape=config.inference.data_source.roi.shape)
    if config.inference.use_database:
        logger.info("input db: %s %s",
                    config.general.db_host,
                    config.inference.data_source.db_name)
        client = pymongo.MongoClient(
            host=config.general.db_host)
        db_name = config.inference.data_source.db_name
        db = client[db_name]
        cells = db[node_collection]
        cell_list = list(cells.find())
        random.shuffle(cell_list)
        num_cells = len(cell_list)
        logger.info("{}, #cells: {}".format(db_name, num_cells))
        voxel_size = gp.Coordinate(config.inference.data_source.voxel_size)
        # scale = np.array(file_resolution)/np.array(voxel_size)
        # logger.info("scaling positions by %s", scale)
        logger.info("using voxel_size: %s", voxel_size)
    else:
        if "polar" in config.inference.data_source.datafile.filename:
            filename_tracks = os.path.splitext(filename_tracks)[0] + "_polar.txt"
        else:
            filename_tracks = os.path.splitext(filename_tracks)[0] + "_div_state.txt"
        logger.info("loading from file %s", filename_tracks)
        voxel_size = np.array(data_config['general']['resolution'])
        logger.info("using voxel_size: %s", voxel_size)

        if "polar" in config.inference.data_source.datafile.filename:
            locations, track_info = parse_tracks_file(
                filename_tracks,
                # scale=scale,
                limit_to_roi=limit_to_roi)

            locationsByClass = {}
            labelsByClass = {}
            for idx, cell in enumerate(track_info):
                locationsByClass.setdefault("polar", []).append(locations[idx])
                labelsByClass.setdefault("polar", []).append(config.model.num_classes)
            logger.info("%s: #class polar: %s",
                        os.path.basename(config.inference.data_source.datafile.filename),
                        len(locationsByClass["polar"]))
        else:
            _, _, locationsByClass, labelsByClass = \
                parse_tracks_file_by_class(
                    filename_tracks,
                    num_classes=config.model.num_classes,
                    # scale=scale,
                    # scale=1.0/np.array(data_config['general']['resolution']),
                    limit_to_roi=limit_to_roi)
        if config.predict.max_samples:
            num_cells = 0
            if "polar" in config.inference.data_source.datafile.filename:
                num_cells = len(locationsByClass["polar"])
            else:
                for cls in range(len(locationsByClass)):
                    locationsByClass[cls], labelsByClass[cls] = \
                        sklearn.utils.shuffle(locationsByClass[cls],
                                              labelsByClass[cls])
                    num_cells += config.predict.max_samples
        else:
            if "polar" in config.inference.data_source.datafile.filename:
                num_cells = len(locationsByClass["polar"])
            else:
                num_cells = 0
                for cls in range(len(locationsByClass)):
                    num_cells += len(locationsByClass[cls])

    raw_key = config.inference.data_source.datafile.group
    if 'nested' in raw_key:
        filename_zarr = zarr.NestedDirectoryStore(filename_zarr)
    data = zarr.open(filename_zarr, 'r')[raw_key]

    logger.info("volume shape: %s", data.shape)
    data_shape = data.shape

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        ({'position': tf.float32, 'input': tf.float32}, tf.int32),
        ({'position': (4,), 'input': input_shape}, tf.TensorShape([])))

    mbs = config.predict.batch_size
    dataset = dataset.repeat(config.predict.test_time_reps).batch(
        mbs).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, num_cells


def model_fn(config, features, labels):

    logger.info("feature tensor: %s", features)
    logger.info("label tensor: %s", labels)

    if config.predict.test_time_reps > 1:
        # keep dropout
        is_training = tf.constant(True, name="is_training")
    else:
        is_training = tf.constant(False, name="is_training")
    raw_batched = tf.reshape(features["input"],
                             [-1] + config.model.input_shape,
                             name="raw_new")
    tf.train.import_meta_graph(
        os.path.join(config.general.setup_dir, "train",
                     config.model.net_name + '.meta'),
        input_map={
            'is_training:0':  is_training,
            'raw:0':  raw_batched},
        clear_devices=True)

    if config.predict.use_swa:
        logger.info("using swa")
        logits = tf.get_default_graph().get_tensor_by_name("swa_model/logits:0")
    else:
        try:
            logits = tf.get_default_graph().get_tensor_by_name("model/logits:0")
        # backwards compatiblity
        except KeyError:
            try:
                logits = tf.get_default_graph().get_tensor_by_name("out/BiasAdd:0")
            except KeyError:
                try:
                    logits = tf.get_default_graph().get_tensor_by_name("conv_fc8_0/BiasAdd:0")
                except KeyError:
                    logits = tf.get_default_graph().get_tensor_by_name("model/out/BiasAdd:0")
            num_classes = (config.model.num_classes + 1
                           if config.model.with_polar
                           else config.model.num_classes)
            logits = tf.reshape(logits, shape=(tf.shape(logits)[0],
                                               num_classes))

    probs = tf.nn.softmax(logits, name="probs")
    cls = tf.argmax(input=logits, axis=1, name="cls",
                    output_type=tf.dtypes.int32)

    predictions = {
        "position": features["position"],
        "class": cls,
        "logits": logits,
        "probabilities": probs,
        "label": labels
    }

    return predictions


def run(config, checkpoint_file):
    sample = config.inference.data_source.datafile.filename
    is_polar = config.model.with_polar

    logger.info("sample: %s (polar? %s)", sample, is_polar)

    if config.inference.use_database:
        node_collection = 'nodes'
        if config.predict.prefix:
            prefix = config.predict.prefix
        else:
            prefix = os.path.join(
                os.path.basename(config.general.setup_dir).split(
                    "classifier_")[-1],
                "test", str(config.inference.checkpoint) + "_")
        assert config.inference.data_source.db_name is not None, \
            "db not found, config {} sample {}".format(
                config.inference.db_meta_info, sample)
    else:
        config.inference.data_source.db_name = \
            "linajea_celegans_" + os.path.basename(config.general.setup_dir)
        if os.path.basename(sample) == "polar":
            name = os.path.basename(os.path.dirname(sample)) + "_" + \
                os.path.basename(sample)
        else:
            name = os.path.basename(sample)
        node_collection = "nodes_" + name +  \
            ("_w_swa" if config.predict.use_swa else "_wo_swa") + \
            ("_ttr{}".format(config.predict.test_time_reps)
             if config.predict.test_time_reps > 1 else "") + \
            ("_roi{}".format(config.inference.data_source.roi.shape[0])
             if config.inference.data_source.roi.shape[0] != 200 else "") + \
            "_" + str(config.inference.checkpoint)
        client = pymongo.MongoClient(host=config.general.db_host)
        db = client[config.inference.data_source.db_name]
        logger.info("checking %s in %s (%s)",
                    node_collection, config.inference.data_source.db_name,
                    db.list_collection_names())
        if node_collection in db.list_collection_names() and \
           not config.inference.force_predict:
            logger.info("already predicted %s in %s, aborting...",
                        node_collection, config.inference.data_source.db_name)
            return
        prefix = config.predict.prefix
    logger.info("output db: %s (prefix: %s)",
                config.inference.data_source.db_name, prefix)

    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(
        config=tf_config)

    dataset, num_cells = cell_division_input_fn(config, sample,
                                                prefix, node_collection)
    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    data_features, data_labels = iterator.get_next(name="iterator_predict")
    sess.run(iterator.initializer)

    predictions = model_fn(config, data_features, data_labels)
    tf.train.Saver().restore(sess, checkpoint_file)

    # to check if batch norm set correctly
    try:
        logger.info("mm %s", sess.run(
            [
            tf.get_default_graph().get_tensor_by_name(
                "model/stack1_block2_0_bn/moving_mean:0"),
            tf.get_default_graph().get_tensor_by_name(
                "model/stack1_block2_0_bn/moving_variance:0")]))
    except:
        try:
            logger.info("mm %s", sess.run(
                [
                    tf.get_default_graph().get_tensor_by_name(
                        "model/conv_10_bn/moving_mean:0"),
                    tf.get_default_graph().get_tensor_by_name(
                        "model/conv_10_bn/moving_variance:0")]))

        except:
            try:
                logger.info("mm %s", sess.run(
                    [
                        tf.get_default_graph().get_tensor_by_name(
                            "batch_normalizateion_1/moving_mean:0"),
                        tf.get_default_graph().get_tensor_by_name(
                            "batch_normalization_1/moving_variance:0")]))
            except:
                try:
                    logger.info("mm %s", sess.run(
                    [
                        tf.get_default_graph().get_tensor_by_name(
                            "model/block2a__bn/moving_mean:0"),
                        tf.get_default_graph().get_tensor_by_name(
                            "model/block2a__bn/moving_variance:0")]))
                except:
                    logger.info("cannot find batch norm stats tensors")

    cnt = 0
    mbs = config.predict.batch_size
    num_batches = num_cells//mbs
    queries = []

    if config.inference.use_database:
        client = pymongo.MongoClient(host=config.general.db_host)
        db = client[config.inference.data_source.db_name]
        cells = db[node_collection]
        logger.info("db meta info %s", db['db_meta_info'].find({}))

    logger.info("doing %s test time repetitions", config.predict.test_time_reps)
    mongodb_op = "$inc" if config.predict.test_time_reps > 1 else "$set"
    factor = 1.0 / config.predict.test_time_reps
    while True:
        try:
            start = time.time()
            p = sess.run(predictions)
            if config.predict.test_time_reps > 1:
                p['probabilities'] *= factor

            time_of_prediction = time.time() - start
            logger.info("batch (sz {}) {}/{}".format(mbs, cnt//mbs, num_batches))
            logger.info("  gt_labels: %s", p['label'])
            logger.info("pred_labels: %s", p['class'])
            if cnt == 0:
                logger.info("pred_probs : %s", p['probabilities'][:5])
                logger.info("pred_logits: %s", p['logits'][:5])

            for idx in range(len(p['label'])):
                logger.debug("{}".format(p['position'][idx]))
                if config.inference.use_database:
                    query = {"t": int(float(p['position'][idx][0])),
                             "z": int(float(p['position'][idx][1])),
                             "y": int(float(p['position'][idx][2])),
                             "x": int(float(p['position'][idx][3]))}
                    update = {mongodb_op: {prefix+"normal":
                                           float(p['probabilities'][idx][0])}}
                    cells.update_one(query, update, upsert=True)
                    update = {mongodb_op: {prefix+"mother":
                                           float(p['probabilities'][idx][1])}}
                    cells.update_one(query, update, upsert=True)
                    update = {mongodb_op: {prefix+"daughter":
                                           float(p['probabilities'][idx][2])}}
                    cells.update_one(query, update, upsert=True)
                    if is_polar:
                        update = {mongodb_op: {prefix+"polar":
                                               float(p['probabilities'][idx][3])}}
                        cells.update_one(query, update, upsert=True)
                else:
                    query = {"t": int(float(p['position'][idx][0])),
                             "z": int(float(p['position'][idx][1])),
                             "y": int(float(p['position'][idx][2])),
                             "x": int(float(p['position'][idx][3])),
                             prefix+"normal":   float(p['probabilities'][idx][0]),
                             prefix+"mother":   float(p['probabilities'][idx][1]),
                             prefix+"daughter": float(p['probabilities'][idx][2])
                             }
                    if is_polar:
                        query[prefix+"polar"] = float(p['probabilities'][idx][3])
                    queries.append(query)
            cnt += len(p['label'])
            time_of_batch = time.time() - start
            logger.info("time pred: {}, batch: {}".format(time_of_prediction,
                                                          time_of_batch))
        except tf.errors.OutOfRangeError:
            logger.info("done predicting")
            break
    logger.info("predicted %s cells", cnt)
    if not config.inference.use_database:
        client = pymongo.MongoClient(host=config.general.db_host)
        db = client[config.inference.data_source.db_name]
        logger.warning("resetting node collection %s in db %s",
                       node_collection, config.inference.data_source.db_name)
        db.drop_collection(node_collection)
        cells = db[node_collection]
        if config.predict.test_time_reps > 1:
            queries_tmp = {}
            for q in queries:
                pos = (q['t'], q['z'], q['y'], q['x'])
                if pos not in queries_tmp:
                    queries_tmp[pos] = q
                else:
                    queries_tmp[pos][prefix+"normal"] += q[prefix+"normal"]
                    queries_tmp[pos][prefix+"mother"] += q[prefix+"mother"]
                    queries_tmp[pos][prefix+"daughter"] += q[prefix+"daughter"]
                    if is_polar:
                        queries_tmp[pos][prefix+"polar"] += q[prefix+"polar"]
            queries = list(queries_tmp.values())
        cells.insert_many(queries)
        logger.info("inserted %s cells into %s", len(queries),
                    config.inference.data_source.db_name)
    client.close()
