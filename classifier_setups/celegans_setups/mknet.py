import json
import logging
import os
import sys

import tensorflow as tf
import numpy as np

# import vgg
# import resnet
from funlib.learn.tensorflow.models import resnet, vgg, EfficientNetBX

logger = logging.getLogger(__name__)


def mk_net(config, output_folder):

    tf.reset_default_graph()
    if config.general.seed:
        tf.set_random_seed(config.general.seed)

    input_shape = config.model.input_shape
    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw_batched = tf.placeholder(tf.float32,
                                 shape=(None,) + input_shape,
                                 name="raw")
    raw_batch = raw_batched
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    # create a U-Net
    if config.model.classify_dataset:
        num_classes = 3
    else:
        num_classes = (config.model.num_classes + 1
                       if config.model.with_polar
                       else config.model.num_classes)
    logger.info("input %s", raw_batched)
    kwargs = {
        'num_classes': num_classes,
        'activation': config.model.activation,
        'padding': config.model.padding,
        'make_iso': config.model.make_isotropic,
        'merge_time_voxel_size': config.model.merge_time_voxel_size,
        'is_training': is_training,
        'use_batchnorm': config.model.use_batchnorm,
        'use_conv4d': config.model.use_conv4d,
        'use_dropout': config.model.use_dropout,
        'voxel_size': config.train_data.data_sources[0].voxel_size
    }

    # he normal
    # initializer = tf.initializers.variance_scaling(
    #     scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None,
    #     dtype=tf.dtypes.float32)
    # glorot normal
    # initializer = tf.initializers.variance_scaling(
    #     scale=1.0, mode='fan_avg', distribution='truncated_normal', seed=None,
    #     dtype=tf.dtypes.float32)
    if config.model.regularizer_weight:
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.model.regularizer_weight)
        logger.info("using regularizer %s %s", regularizer,
                    config.model.regularizer_weight)
    else:
        regularizer = None
    initializer = tf.compat.v1.initializers.he_normal()

    if config.train.use_swa:
        # main model has to be last
        model_scopes = ['swa_model', 'model']
    else:
        model_scopes = ['model']
    for model in model_scopes:
        with tf.variable_scope(model,
                               initializer=initializer,
                               regularizer=regularizer):

            if config.model.network_type == 'resnet':
                logits, network_summaries = resnet(
                    raw_batch,
                    resnet_size=config.model.resnet_size,
                    num_blocks=config.model.num_blocks,
                    use_bottleneck=config.model.use_bottleneck,
                    num_fmaps=config.model.num_fmaps,
                    **kwargs)
            elif config.model.network_type == 'efficientnet':
                logits, network_summaries = EfficientNetBX(
                    raw_batch,
                    efficientnet_size=config.model.efficientnet_size,
                    include_top=not config.model.use_global_pool,
                    **kwargs)
            else:
                logits, network_summaries = vgg(
                    raw_batch,
                    kernel_sizes=config.model.kernel_sizes,
                    fmap_inc_factors=config.model.fmap_inc_factors,
                    downsample_factors=config.model.downsample_factors,
                    fc_size=config.model.fc_size,
                    num_fmaps=config.model.num_fmaps,
                    **kwargs)
        logger.info("logits %s", logits)

    if config.train.use_swa:
        num_models_avgd = tf.get_variable(
            'num_models_avgd',
            shape=(),
            initializer=tf.zeros_initializer,
            trainable=False)
        tvars = tf.trainable_variables()
        model_vars_t = [var for var in tvars
                        if 'swa_model' not in var.name and 'model/' in var.name]
        swa_model_vars_t = [var for var in tvars if 'swa_model' in var.name]
        swa_model_vars = {}
        model_vars = {}
        for v in swa_model_vars_t:
            swa_model_vars[v.name.replace("swa_model", "model")] = v
        for v in model_vars_t:
            model_vars[v.name] = v
        average_ops = []
        average_ops_init = []
        for n, v in swa_model_vars.items():
            average_ops.append(tf.assign(
                swa_model_vars[n],
                swa_model_vars[n] + (model_vars[n] - swa_model_vars[n]) / (num_models_avgd + 1)))
            average_ops_init.append(tf.assign(swa_model_vars[n],
                                              model_vars[n]))
        average_ops = tf.group(average_ops, name='swa_average_op2')
        average_ops_init = tf.group(average_ops_init, name="swa_average_op_init")
        print(average_ops, average_ops_init)

        num_models_avgd_inc = tf.compat.v1.assign_add(num_models_avgd, 1,
                                                      name="swa_model_cnt_inc")
        print(num_models_avgd_inc)

    pred_labels = tf.argmax(input=logits, axis=1, name="classes",
                            output_type=tf.dtypes.int32)
    pred_labels = tf.Print(pred_labels, [pred_labels], message="pred_labels",
                           first_n=5, summarize=512, name="pred_labels")
    logger.info("classes %s", pred_labels)

    pred_probs = tf.nn.softmax(logits, name="softmax_tensor")
    pred_probs = tf.Print(pred_probs, [pred_probs], message="pred_probs",
                          first_n=5, summarize=3*512, name="pred_probs")
    logger.info("probs %s", pred_probs)

    gt_labels = tf.placeholder(tf.int32, shape=[None], name="gt_labels")
    gt_label = tf.Print(gt_labels, [gt_labels], message="  gt_labels",
                        first_n=5, summarize=512, name="gt_labels")
    logger.info("gt labels %s", gt_labels)

    scalar_summaries = []
    # validation_summaries = []
    if config.model.classify_dataset:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels,
                                                      logits=logits)
    elif False:
        weights_class = tf.cast(tf.not_equal(gt_label, 3), dtype=tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels,
                                                      logits=logits,
                                                      weights=weights_class)
        gt_labels_polar = tf.cast(tf.equal(gt_label, 3), dtype=tf.float32)
        _, logits_polar = tf.split(logits, [3, 1], 1)
        print(logits_polar, gt_labels_polar)
        logits_polar = tf.squeeze(logits_polar, axis=1)
        loss_polar = tf.losses.sigmoid_cross_entropy(gt_labels_polar,
                                                     logits=logits_polar)
        scalar_summaries.append(tf.summary.scalar("loss_polar", loss_polar))
        loss += loss_polar
    else:
        if config.model.focal_loss:
            gt_labels_one_hot = tf.one_hot(gt_labels, num_classes)
            weight = 1.0 - tf.reduce_sum(tf.multiply(gt_labels_one_hot,
                                                     pred_probs), axis=1)
            weight = weight*weight
            logger.info("%s %s", gt_labels_one_hot, weight)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels,
                                                          logits=logits,
                                                          weights=weight)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels,
                                                          logits=logits)

    scalar_summaries.append(tf.summary.scalar('loss_ce', loss))

    if config.model.regularizer_weight:
        loss_l2 = tf.losses.get_regularization_loss()
        scalar_summaries.append(tf.summary.scalar('loss_l2', loss_l2))
        loss += loss_l2

    acc = tf.reduce_mean(tf.cast(tf.math.equal(pred_labels, gt_labels),
                                 dtype=tf.float32))
    tp = tf.reduce_sum(tf.cast(tf.math.equal(pred_labels, gt_labels),
                               dtype=tf.float32))
    fp = tf.constant(0, dtype=tf.float32)
    fn = tf.constant(0, dtype=tf.float32)
    acc_c = []
    for c in range(num_classes):
        t2 = tf.math.equal(pred_labels, c)
        t3 = tf.math.equal(gt_labels, c)
        t4 = tf.math.not_equal(pred_labels, c)
        t5 = tf.math.not_equal(gt_labels, c)
        fpc = tf.reduce_sum(tf.cast(tf.math.logical_and(t2, t5),
                                    dtype=tf.float32))
        fnc = tf.reduce_sum(tf.cast(tf.math.logical_and(t4, t3),
                                    dtype=tf.float32))
        # if config.model.classify_dataset:
        tpc = tf.reduce_sum(tf.cast(tf.math.logical_and(t2, t3),
                                    dtype=tf.float32))
        tnc = tf.reduce_sum(tf.cast(tf.math.logical_and(t4, t5),
                                    dtype=tf.float32))
        acc_c.append((tpc+tnc)/(tpc+tnc+fpc+fnc))
        scalar_summaries.append(tf.summary.scalar('acc_'+str(c), acc_c[-1]))

        fp += fpc
        fn += fnc
    ap = tp / (tp+fp+fn)
    if config.model.classify_dataset:
        acc = tf.Print(acc, [acc, *acc_c, pred_labels, gt_labels],
                       message='acc', summarize=1024)
    scalar_summaries.append(tf.summary.scalar('loss', loss))
    scalar_summaries.append(tf.summary.scalar('acc', acc))
    scalar_summaries.append(tf.summary.scalar('ap', ap))

    scalar_summaries = tf.summary.merge(scalar_summaries,
                                        name="scalar_summaries")
    network_summaries = tf.summary.merge(network_summaries,
                                         name="network_summaries")

    image_summaries = []
    image_summaries.append(
        tf.summary.image("gt_labels",
                         tf.cast(tf.reshape(gt_label, (1, -1, 1, 1)),
                                 tf.uint8)*85,
                         max_outputs=1))
    image_summaries.append(
        tf.summary.image("pred_labels",
                         tf.cast(tf.reshape(pred_labels, (1, -1, 1, 1)),
                                 tf.uint8)*85,
                         max_outputs=1))
    image_summaries.append(
        tf.summary.image("pred_probs",
                         tf.reshape(pred_probs, (1, -1, num_classes, 1)),
                         max_outputs=1))
    image_summaries = tf.summary.merge(image_summaries, name="image_summaries")

    logger.info("optimizer %s", config.optimizerTF1.optimizer)
    if config.optimizerTF1.optimizer == "AdamWOptimizer":
        opt = getattr(tf.contrib.opt, config.optimizerTF1.optimizer)(
            *config.optimizerTF1.get_args(),
            **config.optimizerTF1.get_kwargs())
    else:
        opt = getattr(tf.train, config.optimizerTF1.optimizer)(
            *config.optimizerTF1.get_args(),
            **config.optimizerTF1.get_kwargs())
    logger.info("optimizer: %s", opt)

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)
    tvars = tf.trainable_variables()
    model_vars = [var for var in tvars if 'swa_model' not in var.name]
    optimizer = opt.minimize(loss=loss,
                             var_list=model_vars,
                             global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([optimizer, update_ops])

    fn = os.path.join(output_folder, config.model.net_name)

    tf.train.export_meta_graph(filename=fn + '.meta')

    names = {
        'raw': raw_batched.name,
        'gt_labels': gt_labels.name,
        'pred_labels': pred_labels.name,
        'pred_probs': pred_probs.name,
        'loss': loss.name,
        'optimizer': train_op.name,
        'scalar_summaries': scalar_summaries.name,
        # 'validation_summaries': validation_summaries.name,
        'network_summaries': network_summaries.name,
        'image_summaries': image_summaries.name,
        'is_training': is_training.name
    }

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    net_config = {
        'input_shape': input_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(net_config, f)
