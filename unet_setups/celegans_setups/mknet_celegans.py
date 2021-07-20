import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import json
import logging
import os
import sys

import tensorflow as tf
import toml

from funlib.learn.tensorflow.models import conv_pass, crop, unet
from linajea import load_config


def create_network(config, input_shape, setup_dir, name):

    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    name = os.path.join(setup_dir, name)

    tf.reset_default_graph()

    # l, d, h, w
    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw")

    # b=1, c=1, l, d, h, w
    raw_batched = tf.reshape(raw, [1, 1] + input_shape)

    with tf.variable_scope('parent_vectors'):

        # b=1, c=12, d, h, w
        out, _, _ = unet(
            raw_batched,
            config['model']['num_fmaps'],
            config['model']['fmap_inc_factors'],
            config['model']['downsample_factors'],
            kernel_size_down=config['model']['kernel_size_down'],
            kernel_size_up=config['model']['kernel_size_up'],
            upsampling="resize_conv",
            voxel_size=config['data']['voxel_size'][1:])
        # b=1, c=3, d, h, w
        parent_vectors_batched, _ = conv_pass(
            out,
            kernel_sizes=[1],
            num_fmaps=3,
            activation=None,
            name='parent_vectors')

    with tf.variable_scope('cell_indicator'):

        # b=1, c=12, d, h, w
        out, _, _ = unet(
            raw_batched,
            config['model']['num_fmaps'],
            config['model']['fmap_inc_factors'],
            config['model']['downsample_factors'],
            kernel_size_down=config['model']['kernel_size_down'],
            kernel_size_up=config['model']['kernel_size_up'],
            upsampling="resize_conv",
            voxel_size=config['data']['voxel_size'][1:])

        # b=1, c=1, d, h, w
        cell_indicator_batched, _ = conv_pass(
            out,
            kernel_sizes=[1],
            num_fmaps=1,
            activation='sigmoid',
            name='cell_indicator')

    # there are outputs of two sizes:
    # 1. the prediction output
    # 2. the NMS output, smaller than prediction due to VALID max_pool

    # l=1, d, h, w
    output_shape_1 = tuple(cell_indicator_batched.get_shape().as_list()[1:])
    # c=3, l=1, d, h, w
    parent_vectors = tf.reshape(parent_vectors_batched, (3,) + output_shape_1)
    gt_parent_vectors = tf.placeholder(tf.float32, shape=(3,) + output_shape_1,
                                       name="gt_parent_vectors")

    # l=1, d, h, w
    cell_indicator = tf.reshape(cell_indicator_batched, output_shape_1)
    gt_cell_indicator = tf.placeholder(tf.float32, shape=output_shape_1,
                                       name="gt_cell_indicator")

    # l=1, d, h, w
    cell_mask = tf.placeholder(tf.bool, shape=output_shape_1,
                               name="gt_cell_mask")

    # radius of about [10, 10, 10] at voxel size [5, 1, 1]
    # has to be odd
    nms_window_shape = config['model']['nms_window_shape']

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

    # cropped
    parent_vectors_loss_maxima = tf.losses.mean_squared_error(
        # c=3, l=1, d', h', w'
        gt_parent_vectors_cropped,
        # c=3, l=1, d', h', w'
        parent_vectors_cropped,
        # c=1, l=1, d', h', w' (broadcastable)
        tf.reshape(maxima_in_cell_mask, (1,) + output_shape_2))

    # non-cropped
    cond = tf.less(gt_cell_indicator, 0.01)
    weight = tf.where(cond,
                      tf.constant(0.00001, dtype=tf.float32,
                                  shape=cell_indicator.get_shape()),
                      tf.constant(1, dtype=tf.float32,
                                  shape=cell_indicator.get_shape()))
    cell_indicator_loss = tf.losses.mean_squared_error(
        # l=1, d, h, w
        gt_cell_indicator,
        # l=1, d, h, w
        cell_indicator,
        # l=1, d, h, w
        weight)
        # cell_mask)

    opt = tf.train.AdamOptimizer(
        learning_rate=config['optimizer']['lr'],
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)

    # smooth transition from training parent vectors on complete cell mask to
    # only on maxima
    # https://www.wolframalpha.com/input/?i=1.0%2F(1.0+%2B+exp(0.01*(-x%2B20000)))+x%3D0+to+40000
    alpha = tf.constant(1.0)/(
        tf.constant(1.0) + tf.exp(
            tf.constant(0.0005) *
            (-iteration +
             tf.cast(tf.constant(config['training']['parent_vectors_loss_transition']),
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

    summary = tf.summary.merge([
        tf.summary.scalar('parent_vectors_loss_maxima',
                          parent_vectors_loss_maxima),
        tf.summary.scalar('parent_vectors_loss_cell_mask',
                          parent_vectors_loss_cell_mask),
        tf.summary.scalar('cell_indicator_loss', cell_indicator_loss),
        tf.summary.scalar('alpha', alpha),
        tf.summary.scalar('loss', loss)])

    optimizer = opt.minimize(loss, global_step=iteration)

    # l=1, d, h, w
    print("input shape : %s" % (input_shape,))
    print("output shape 1: %s" % (output_shape_1,))
    print("output shape 2: %s" % (output_shape_2,))

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
        'cell_mask': cell_mask.name,
        'maxima': maxima.name,
        'maxima_in_cell_mask': maxima_in_cell_mask.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'summary': summary.name,
        'input_shape': input_shape,
        'output_shape_1': output_shape_1,
        'output_shape_2': output_shape_2}
    with open(name + '_config.json', 'w') as f:
        json.dump(names, f)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Number of parameters:", total_parameters)
    print("Estimated size of parameters in GB:",
          float(total_parameters)*8/(1024*1024*1024))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--setup_dir', type=str,
                        required=True, help='output')

    args = parser.parse_args()
    config = load_config(args.config)
    os.makedirs(args.setup_dir, exist_ok=True)

    create_network(config, config['model']['train_input_shape'],
                   args.setup_dir, 'train_net')
    create_network(config, config['model']['predict_input_shape'],
                   args.setup_dir, 'test_net')
