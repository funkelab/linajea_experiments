import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import json
import logging
import os
import sys

import attr
import tensorflow as tf
import toml

from funlib.learn.tensorflow.models import conv_pass, crop, unet, crop_to_factor
from linajea.config import TrackingConfig

logger = logging.getLogger(__name__)

def create_network(input_shape, name, config):

    name = os.path.join(config.general.setup_dir, name)
    voxel_size = config.train_data.data_sources[0].voxel_size[1:]

    tf.reset_default_graph()

    # l, d, h, w
    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw")

    # b=1, c=1, l, d, h, w
    raw_batched = tf.reshape(raw, [1, 1] + input_shape)

    initializer = tf.compat.v1.initializers.he_normal()
    if config.model.unet_style == "single":
        with tf.variable_scope('unet',
                               initializer=initializer):

            # b=1, c=12, d, h, w
            out, _, _ = unet(
                raw_batched,
                config.model.num_fmaps,
                config.model.fmap_inc_factors,
                config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                upsampling="resize_conv",
                voxel_size=voxel_size)

            # b=1, c=3, d, h, w
            parent_vectors_batched, _ = conv_pass(
                out,
                kernel_sizes=[1],
                num_fmaps=3,
                activation=None,
                name='parent_vectors')
            # b=1, c=1, d, h, w
            cell_indicator_batched, _ = conv_pass(
                out,
                kernel_sizes=[1],
                num_fmaps=1,
                activation='sigmoid',
                name='cell_indicator')

            if config.train.cell_density:
                if isinstance(config.train.cell_density, int):
                    num_fmaps_cell_den = config.train.cell_density
                else:
                    num_fmaps_cell_den = 1
                # b=1, c=1, d, h, w
                cell_density_batched, _ = conv_pass(
                    out,
                    kernel_sizes=[1],
                    num_fmaps=num_fmaps_cell_den,
                    activation=None,
                    name='cell_density')
    elif config.model.unet_style == "multihead":
        with tf.variable_scope('unet_multihead',
                               initializer=initializer):

            if config.train.cell_density:
                num_heads = 3
            else:
                num_heads = 2

            # b=1, c=12, d, h, w
            out, _, _ = unet(
                raw_batched,
                config.model.num_fmaps,
                config.model.fmap_inc_factors,
                config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                upsampling="resize_conv",
                voxel_size=voxel_size,
                num_heads=num_heads)

            # b=1, c=3, d, h, w
            parent_vectors_batched, _ = conv_pass(
                out[0],
                kernel_sizes=[1],
                num_fmaps=3,
                activation=None,
                name='parent_vectors')
            # b=1, c=1, d, h, w
            cell_indicator_batched, _ = conv_pass(
                out[1],
                kernel_sizes=[1],
                num_fmaps=1,
                activation='sigmoid',
                name='cell_indicator')

            if config.train.cell_density:
                if isinstance(config.train.cell_density, int):
                    num_fmaps_cell_den = config.train.cell_density
                else:
                    num_fmaps_cell_den = 1
                # b=1, c=1, d, h, w
                cell_density_batched, _ = conv_pass(
                    out[2],
                    kernel_sizes=[1],
                    num_fmaps=num_fmaps_cell_den,
                    activation=None,
                    name='cell_density')
    elif config.model.unet_style is None or \
         config.model.unet_style == "split":
        with tf.variable_scope('parent_vectors',
                               initializer=initializer):

            if isinstance(config.model.num_fmaps, list):
                num_fmaps = config.model.num_fmaps
            else:
                num_fmaps = 3*[config.model.num_fmaps]

            # b=1, c=12, d, h, w
            out, _, _ = unet(
                raw_batched,
                num_fmaps[0],
                config.model.fmap_inc_factors,
                config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                upsampling="resize_conv",
                voxel_size=voxel_size)
            # b=1, c=3, d, h, w
            parent_vectors_batched, _ = conv_pass(
                out,
                kernel_sizes=[1],
                num_fmaps=3,
                activation=None,
                name='parent_vectors')

        with tf.variable_scope('cell_indicator',
                               initializer=initializer):

            # b=1, c=12, d, h, w
            out, _, _ = unet(
                raw_batched,
                num_fmaps[1],
                config.model.fmap_inc_factors,
                config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                upsampling="resize_conv",
                voxel_size=voxel_size)

            # b=1, c=1, d, h, w
            cell_indicator_batched, _ = conv_pass(
                out,
                kernel_sizes=[1],
                num_fmaps=1,
                activation='sigmoid',
                name='cell_indicator')

        if config.train.cell_density:
            with tf.variable_scope('cell_density',
                                   initializer=initializer):

                # b=1, c=12, d, h, w
                out, _, _ = unet(
                    raw_batched,
                    num_fmaps[2],
                    config.model.fmap_inc_factors,
                    config.model.downsample_factors,
                    kernel_size_down=config.model.kernel_size_down,
                    kernel_size_up=config.model.kernel_size_up,
                    upsampling="resize_conv",
                    voxel_size=voxel_size)

                if isinstance(config.train.cell_density, int):
                    num_fmaps_cell_den = config.train.cell_density
                else:
                    num_fmaps_cell_den = 1
                # b=1, c=1, d, h, w
                cell_density_batched, _ = conv_pass(
                    out,
                    kernel_sizes=[1],
                    num_fmaps=num_fmaps_cell_den,
                    activation=None,
                    name='cell_density')
    else:
        raise RuntimeError("invalid unet_syle {}".format(config.model.unet_style))

    # there are outputs of two sizes:
    # 1. the prediction output
    # 2. the NMS output, smaller than prediction due to VALID max_pool

    # l=1, d, h, w
    output_shape_1 = tuple(cell_indicator_batched.get_shape().as_list()[1:])
    # c=3, l=1, d, h, w
    parent_vectors = tf.reshape(parent_vectors_batched,
                                (3,) + output_shape_1)
    gt_parent_vectors = tf.placeholder(tf.float32,
                                       shape=(3,) + output_shape_1,
                                       name="gt_parent_vectors")

    # l=1, d, h, w
    cell_indicator = tf.reshape(cell_indicator_batched,
                                output_shape_1)
    gt_cell_indicator = tf.placeholder(tf.float32, shape=output_shape_1,
                                       name="gt_cell_indicator")

    gt_cell_center = tf.placeholder(tf.float32, shape=output_shape_1,
                                    name="gt_cell_center")

    # l=1, d, h, w
    cell_mask = tf.placeholder(tf.bool, shape=output_shape_1,
                               name="gt_cell_mask")

    anchor = tf.placeholder(tf.float32, shape=output_shape_1,
                            name="anchor")

    if config.train.cell_density:
        # l=1, d, h, w
        cell_density = tf.squeeze(cell_density_batched, axis=0)
        gt_cell_density = tf.placeholder(tf.int32, shape=output_shape_1,
                                         name="gt_cell_density")

    # radius of about [10, 10, 10] at voxel size [5, 1, 1]
    # has to be odd
    if os.path.basename(name) == "test_net" \
       and config.model.nms_window_shape_test is not None:
        nms_window_shape = config.model.nms_window_shape_test
        logger.info("nms window shape %s", nms_window_shape)
    else:
        nms_window_shape = config.model.nms_window_shape

    # b=1, c=1, d', h', w' (with d'=d-k_d+1, same for h', w'; k = window shape)
    maxima = tf.nn.pool(
        # b=1, c=1, d, h, w
        cell_indicator_batched,
        nms_window_shape,
        'MAX',
        'VALID',
        strides=[1, 1, 1],
        data_format='NCDHW'
    )

    # wrong crop: comment
    if os.path.basename(name) == "test_net":
        logger.info("maxima uncropped: %s",
                    maxima.shape)
        factor_product = None
        for factor in config.model.downsample_factors:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(
                    f*ff
                    for f, ff in zip(factor, factor_product))
        maxima = crop_to_factor(
            maxima,
            factor=factor_product,
            kernel_sizes=[1])

        logger.info("maxima cropped: %s",
                    maxima.shape)

    # l=1, d', h', w'
    output_shape_2 = tuple(maxima.get_shape().as_list()[1:])

    # l=1, d', h', w'
    maxima = tf.reshape(maxima, output_shape_2)

    raw_cropped = crop(raw, output_shape_1)

    # l=1, d', h', w'
    cell_indicator_cropped = crop(
        # l=1, d, h, w
        cell_indicator,
        # l=1, d', h', w'
        output_shape_2)

    # l=1, d', h', w'
    maxima = tf.equal(
        cell_indicator_cropped,
        maxima)

    # l=1, d', h', w'
    cell_mask_cropped = crop(
        # l=1, d, h, w
        cell_mask,
        # l=1, d', h', w'
        output_shape_2)

    # l=1, d', h', w'
    maxima_in_cell_mask = tf.logical_and(maxima, cell_mask_cropped)
    maxima_in_cell_mask = tf.reshape(maxima_in_cell_mask,
                                     (1,) + output_shape_2)

    # c=3, l=1, d', h', w'
    parent_vectors_cropped = crop(
        # c=3, l=1, d, h, w
        parent_vectors,
        # c=3, l=1, d', h', w'
        (3,) + output_shape_2)

    # c=3, l=1, d', h', w'
    gt_parent_vectors_cropped = crop(
        # c=3, l=1, d, h, w
        gt_parent_vectors,
        # c=3, l=1, d', h', w'
        (3,) + output_shape_2)

    # losses for training on non-cropped outputs, if possible:

    # non-cropped
    parent_vectors_loss_cell_mask = tf.losses.mean_squared_error(
        # c=3, l=1, d, h, w
        gt_parent_vectors,
        # c=3, l=1, d, h, w
        parent_vectors,
        # c=1, l=1, d, h, w (broadcastable)
        tf.reshape(cell_mask, (1,) + output_shape_1))
        # tf.reshape(cell_mask, (1,) + output_shape_1))

    # cropped
    parent_vectors_loss_maxima = tf.losses.mean_squared_error(
        # c=3, l=1, d', h', w'
        gt_parent_vectors_cropped,
        # c=3, l=1, d', h', w'
        parent_vectors_cropped,
        # c=1, l=1, d', h', w' (broadcastable)
        tf.reshape(maxima_in_cell_mask, (1,) + output_shape_2))

    # non-cropped
    if config.model.cell_indicator_weighted:
        cond = tf.less(gt_cell_indicator, 0.01)
        weight = tf.where(cond,
                          tf.constant(0.00001, dtype=tf.float32,
                                      shape=cell_indicator.get_shape()),
                          tf.constant(1, dtype=tf.float32,
                                      shape=cell_indicator.get_shape()))
    else:
        weight = 1
    cell_indicator_loss = tf.losses.mean_squared_error(
        # l=1, d, h, w
        gt_cell_indicator,
        # l=1, d, h, w
        cell_indicator,
        # l=1, d, h, w
        weight)
        # cell_mask)

    if config.train.cell_density:
        # isinstance(True, int) -> True
        if isinstance(config.train.cell_density, bool):
            logger.info("mse loss %s %s", gt_cell_density, cell_density)
            cell_density_loss = tf.losses.mean_squared_error(
                # l=1, d, h, w
                gt_cell_density,
                # l=1, d, h, w
                cell_density)
        else:
            logger.info("cross entropy loss %s %s", gt_cell_density, cell_density)
            cell_density_loss = tf.losses.sparse_softmax_cross_entropy(
                # l=1, d, h, w
                tf.squeeze(gt_cell_density, axis=0),
                # l=1, d, h, w
                tf.transpose(cell_density, [1, 2, 3, 0]))
            cell_density = tf.dtypes.cast(
                tf.expand_dims(tf.math.argmax(cell_density, axis=0,
                                              output_type=tf.int32), axis=0),
                tf.uint8)

    opt = getattr(tf.train, config.optimizerTF1.optimizer)(
        *config.optimizerTF1.get_args(),
        **config.optimizerTF1.get_kwargs())
    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)


    if config.train.parent_vectors_loss_transition_offset:
        # smooth transition from training parent vectors on complete cell mask to
        # only on maxima
        # https://www.wolframalpha.com/input/?i=1.0%2F(1.0+%2B+exp(0.01*(-x%2B20000)))+x%3D0+to+40000
        alpha = tf.constant(1.0)/(
            tf.constant(1.0) + tf.exp(
                tf.constant(config.train.parent_vectors_loss_transition_factor) *
                (-iteration +
                 tf.cast(tf.constant(config.train.parent_vectors_loss_transition_offset),
                         tf.float32))
            )
        )

        # multiply cell indicator loss with parent vector loss, since they have
        # different magnitudes (this normalizes gradients by magnitude of other
        # loss: (uv)' = u'v + uv')
        loss = cell_indicator_loss + (
            parent_vectors_loss_maxima*alpha +
            parent_vectors_loss_cell_mask*(1.0 - alpha)
        )
    else:
        loss = cell_indicator_loss + parent_vectors_loss_cell_mask
    if config.train.cell_density:
        loss += cell_density_loss

    # loss = cell_indicator_loss
    scalar_summaries = [
        tf.summary.scalar('parent_vectors_loss_maxima',
                          parent_vectors_loss_maxima),
        tf.summary.scalar('parent_vectors_loss_cell_mask',
                          parent_vectors_loss_cell_mask),
        tf.summary.scalar('cell_indicator_loss', cell_indicator_loss),
        tf.summary.scalar('loss', loss)]
    if config.train.parent_vectors_loss_transition_offset:
        scalar_summaries.append(
            tf.summary.scalar('alpha', alpha))
    if config.train.cell_density:
        scalar_summaries.append(
            tf.summary.scalar('cell_density_loss', cell_density_loss))
    scalar_summaries = tf.summary.merge(scalar_summaries)
    metric_summaries = add_metric_summaries(gt_cell_center,
                                            cell_indicator,
                                            cell_indicator_cropped,
                                            gt_parent_vectors_cropped,
                                            parent_vectors_cropped,
                                            maxima_in_cell_mask,
                                            output_shape_2,
                                            voxel_size)

    optimizer = opt.minimize(loss, global_step=iteration)

    # l=1, d, h, w
    logger.info("input shape : %s" % (input_shape,))
    logger.info("output shape 1: %s" % (output_shape_1,))
    logger.info("output shape 2: %s" % (output_shape_2,))

    tf.train.export_meta_graph(filename=name + '.meta')

    names = {
        'raw': raw.name,
        'raw_cropped': raw_cropped.name,
        'parent_vectors': parent_vectors.name,
        'parent_vectors_cropped': parent_vectors_cropped.name,
        'gt_parent_vectors': gt_parent_vectors.name,
        'cell_indicator': cell_indicator.name,
        'cell_indicator_cropped': cell_indicator_cropped.name,
        'gt_cell_indicator': gt_cell_indicator.name,
        'gt_cell_center': gt_cell_center.name,
        'cell_mask': cell_mask.name,
        'anchor': anchor.name,
        'maxima': maxima.name,
        'maxima_in_cell_mask': maxima_in_cell_mask.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'scalar_summaries': scalar_summaries.name,
        'metric_summaries': metric_summaries.name
    }
    if config.train.cell_density:
        names['cell_density'] = cell_density.name
        names['gt_cell_density'] = gt_cell_density.name
    with open(name + '_names.json', 'w') as f:
        json.dump(names, f)

    net_config = {
        'input_shape': input_shape,
        'output_shape_1': output_shape_1,
        'output_shape_2': output_shape_2
    }
    with open(name + '_config.json', 'w') as f:
        json.dump(net_config, f)


    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    logger.info("Number of parameters: %s", total_parameters)
    logger.info("Estimated size of parameters in GB: %s",
                float(total_parameters)*8/(1024*1024*1024))


def add_metric_summaries(gt_cell_center,
                         cell_indicator,
                         cell_indicator_cropped,
                         gt_parent_vectors_cropped,
                         parent_vectors_cropped,
                         maxima_in_cell_mask,
                         output_shape_2,
                         voxel_size):
    first_n = 10

    # ground truth cell locations
    gt_max_loc = tf.where(gt_cell_center > 0.5)
    gt_max_loc = tf.Print(gt_max_loc,
                          [gt_max_loc, tf.reduce_max(gt_cell_center)],
                          message="gt_max_loc_ind", first_n=first_n,
                          summarize=512, name="gt_max_loc")

    # predicted value at those locations
    tmp = tf.gather_nd(cell_indicator, gt_max_loc)
    # true positive if > 0.5
    cell_ind_tpr = tf.reduce_mean(tf.cast(tf.math.greater(tmp, 0.5),
                                          tf.float32))
    cell_ind_tpr_gt = tf.Print(cell_ind_tpr, [cell_ind_tpr],
                               message="cell_ind_tpr", first_n=first_n,
                               summarize=512, name="cell_ind_tpr")

    # crop to nms area
    gt_cell_center_cropped = crop(
        # l=1, d, h, w
        gt_cell_center,
        # l=1, d', h', w'
        output_shape_2)
    tp_dims = [1, 2, 3, 4, 0]

    # cropped ground truth cell locations
    gt_max_loc = tf.where(gt_cell_center_cropped > 0.5)
    gt_max_loc = tf.Print(gt_max_loc, [gt_max_loc],
                          message="gt_max_loc", first_n=first_n,
                          summarize=512, name="gt_max_loc")

    # ground truth parent vectors at those locations
    tmp_gt_par = tf.gather_nd(tf.transpose(
        gt_parent_vectors_cropped, tp_dims), gt_max_loc)
    tmp_gt_par = tf.Print(tmp_gt_par, [tmp_gt_par],
                          message="gt_par_vec", first_n=first_n,
                          summarize=512, name="gt_par_vec")

    # predicted parent vectors at those locations
    tmp_par = tf.gather_nd(tf.transpose(
        parent_vectors_cropped, tp_dims), gt_max_loc)
    tmp_par = tf.Print(tmp_par, [tmp_par],
                       message="pred_par_vec", first_n=first_n,
                       summarize=512, name="pred_par_vec")

    # normalize predicted parent vectors
    normalize_pred = tf.math.l2_normalize(tmp_par, 1)
    normalize_pred = tf.Print(normalize_pred, [normalize_pred],
                              message="norm_pred_par_vec", first_n=first_n,
                              summarize=512, name="norm_pred_par_vec")
    # normalize ground truth parent vectors
    normalize_gt = tf.math.l2_normalize(tmp_gt_par, 1)
    normalize_gt = tf.Print(normalize_gt, [normalize_gt],
                            message="norm_gt_par_vec", first_n=first_n,
                            summarize=512, name="norm_gt_par_vec")

    # cosine similarity predicted vs ground truth parent vectors
    cos_similarity=tf.math.reduce_sum(tf.multiply(normalize_pred,normalize_gt),
                                      axis=1)
    cos_similarity = tf.Print(cos_similarity, [cos_similarity],
                              message="cos_similarity_par_vec", first_n=first_n,
                              summarize=512, name="cos_similarity_par_vec")

    # rate with cosine similarity > 0.9
    par_vec_cos = tf.reduce_mean(tf.cast(tf.math.greater(cos_similarity, 0.9),
                                         tf.float32))
    par_vec_cos_gt = tf.Print(par_vec_cos, [par_vec_cos],
                              message="par_vec_cos", first_n=first_n,
                              summarize=512, name="par_vec_cos")


    # distance between endpoints of predicted vs gt parent vectors
    par_vec_diff = tf.norm(tf.math.subtract(
        tf.divide(tmp_gt_par, voxel_size),
        tf.divide(tmp_par, voxel_size)), axis=1)
    par_vec_diff = tf.Print(par_vec_diff,
                            [par_vec_diff,
                             tf.reduce_min(par_vec_diff),
                             tf.reduce_max(par_vec_diff)],
                            message="par_vec_diff", first_n=first_n,
                            summarize=512, name="par_vec_diff")
    # mean distance
    par_vec_diff_mn_gt = tf.reduce_mean(par_vec_diff)

    # rate with distance < 1
    par_vec_tpr = tf.reduce_mean(tf.cast(tf.math.less(par_vec_diff, 1),
                                         tf.float32))
    par_vec_tpr_gt = tf.Print(par_vec_tpr, [par_vec_tpr],
                              message="par_vec_tpr", first_n=first_n,
                              summarize=512, name="par_vec_tpr")


    # predicted cell locations
    pred_max_loc = tf.where(tf.reshape(maxima_in_cell_mask, output_shape_2))
    pred_max_loc = tf.Print(pred_max_loc, [pred_max_loc],
                            message="pred_max_loc", first_n=first_n,
                            summarize=512, name="pred_max_loc")

    # predicted value at those locations
    tmp = tf.gather_nd(cell_indicator_cropped, pred_max_loc)
    # assumed good if > 0.5
    cell_ind_tpr = tf.reduce_mean(tf.cast(tf.math.greater(tmp, 0.5),
                                          tf.float32))
    cell_ind_tpr_pred = tf.Print(cell_ind_tpr, [cell_ind_tpr],
                                 message="cell_ind_tpr", first_n=first_n,
                                 summarize=512, name="cell_ind_tpr")


    tp_dims = [1, 2, 3, 4, 0]

    # ground truth parent vectors at those locations
    tmp_gt_par = tf.gather_nd(tf.transpose(
        gt_parent_vectors_cropped, tp_dims), pred_max_loc)
    tmp_gt_par = tf.Print(tmp_gt_par, [tmp_gt_par],
                          message="gt_par_vec", first_n=first_n,
                          summarize=512, name="gt_par_vec")

    # predicted parent vectors at those locations
    tmp_par = tf.gather_nd(tf.transpose(
        parent_vectors_cropped, tp_dims), pred_max_loc)
    tmp_par = tf.Print(tmp_par, [tmp_par],
                       message="pred_par_vec", first_n=first_n,
                       summarize=512, name="pred_par_vec")

    # normalize predicted parent vectors
    normalize_pred = tf.math.l2_normalize(tmp_par, 1)
    normalize_pred = tf.Print(normalize_pred, [normalize_pred],
                              message="norm_pred_par_vec", first_n=first_n,
                              summarize=512, name="norm_pred_par_vec")

    # normalize ground truth parent vectors
    normalize_gt = tf.math.l2_normalize(tmp_gt_par, 1)
    normalize_gt = tf.Print(normalize_gt, [normalize_gt],
                            message="norm_gt_par_vec",
                            first_n=first_n, summarize=512, name="norm_gt_par_vec")

    # cosine similarity predicted vs ground truth parent vectors
    cos_similarity=tf.math.reduce_sum(tf.multiply(normalize_pred,normalize_gt),
                                      axis=1)
    cos_similarity = tf.Print(cos_similarity, [cos_similarity],
                              message="cos_similarity_par_vec", first_n=first_n,
                              summarize=512, name="cos_similarity_par_vec")

    # rate with cosine similarity > 0.9
    par_vec_cos = tf.reduce_mean(tf.cast(tf.math.greater(cos_similarity, 0.9),
                                         tf.float32))
    par_vec_cos_pred = tf.Print(par_vec_cos, [par_vec_cos],
                                message="par_vec_cos", first_n=first_n,
                                summarize=512, name="par_vec_cos")


    # distance between endpoints of predicted vs gt parent vectors
    par_vec_diff = tf.norm(tf.math.subtract(
        tf.divide(tmp_gt_par, voxel_size),
        tf.divide(tmp_par, voxel_size)), axis=1)
    par_vec_diff = tf.Print(par_vec_diff,
                            [par_vec_diff,
                             tf.reduce_min(par_vec_diff),
                             tf.reduce_max(par_vec_diff)],
                            message="par_vec_diff", first_n=first_n,
                            summarize=512, name="par_vec_diff")

    # mean distance
    par_vec_diff_mn_pred = tf.reduce_mean(par_vec_diff)

    # rate with distance < 1
    par_vec_tpr = tf.reduce_mean(tf.cast(tf.math.less(par_vec_diff, 1),
                                         tf.float32))
    par_vec_tpr_pred = tf.Print(par_vec_tpr, [par_vec_tpr],
                                message="par_vec_tpr", first_n=first_n,
                                summarize=512, name="par_vec_tpr")

    metric_summaries = [
        tf.summary.scalar('cell_ind_tpr_gt', cell_ind_tpr_gt),
        tf.summary.scalar('par_vec_cos_gt', par_vec_cos_gt),
        tf.summary.scalar('par_vec_diff_gt', par_vec_diff_mn_gt),
        tf.summary.scalar('par_vec_tpr_gt', par_vec_tpr_gt),
        tf.summary.scalar('cell_ind_tpr_pred', cell_ind_tpr_pred),
        tf.summary.scalar('par_vec_cos_pred', par_vec_cos_pred),
        tf.summary.scalar('par_vec_diff_pred', par_vec_diff_mn_pred),
        tf.summary.scalar('par_vec_tpr_pred', par_vec_tpr_pred)
    ]
    metric_summaries = tf.summary.merge(metric_summaries)

    return metric_summaries


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
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    create_network(config.model.train_input_shape, 'train_net',
                   config)
    create_network(config.model.predict_input_shape, 'test_net',
                   config)
