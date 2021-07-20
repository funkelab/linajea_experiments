import tensorflow as tf
import json
from funlib.learn.tensorflow.models.unet import unet, conv_pass, crop_to_factor
import logging
import argparse
import os

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def crop(a, shape):
    '''Crop a to a new shape, centered in a.

    Args:

        a:

            The input tensor.

        shape:

            A list (not a tensor) with the requested shape.
    '''

    in_shape = a.get_shape().as_list()

    offset = list([
        (i - s)//2
        for i, s in zip(in_shape, shape)
    ])

    b = tf.slice(a, offset, shape)

    return b


def create_network(
        input_shape,
        name,
        constant,
        average,
        setup_dir):
    if not os.path.isdir(setup_dir):
        os.mkdir(setup_dir)
    tf.reset_default_graph()

    # c=2, l, d, h, w
    raw = tf.placeholder(tf.float32, shape=(2,) + input_shape)
    # b=1, c=2, l, d, h, w
    raw_batched = tf.reshape(raw, (1, 2) + input_shape)

    with tf.variable_scope('parent_vectors'):
        downsample_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        # b=1, c=12, d, h, w
        out, _, _ = unet(
                raw_batched, 12, 3,
                downsample_factors,
                voxel_size=(3, 3, 3),
                constant_upsample=constant)

        # b=1, c=3, d, h, w
        parent_vectors_batched, _ = conv_pass(
            out,
            kernel_sizes=[1], num_fmaps=3,
            activation=None,
            name='parent_vectors')

    with tf.variable_scope('cell_indicator'):

        # b=1, c=12, d, h, w
        out, _, _ = unet(
                raw_batched, 12, 3,
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                voxel_size=(3, 3, 3),
                constant_upsample=constant)

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
    gt_parent_vectors = tf.placeholder(tf.float32, shape=(3,) + output_shape_1)

    # l=1, d, h, w
    cell_indicator = tf.reshape(cell_indicator_batched, output_shape_1)
    gt_cell_indicator = tf.placeholder(tf.float32, shape=output_shape_1)

    # l=1, d, h, w
    cell_mask = tf.placeholder(tf.bool, shape=output_shape_1)

    # radius of about [10, 10, 10] at voxel size [3, 3, 3]
    # has to be odd
    nms_window_shape = [7, 7, 7]

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

    if os.path.basename(name) == "test_net":
        factor_product = None
        for factor in downsample_factors:
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

    # l=1, d', h', w'
    output_shape_2 = tuple(maxima.get_shape().as_list()[1:])

    # l=1, d', h', w'
    maxima = tf.reshape(maxima, output_shape_2)

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

    if average:
        # Expand maxima to 3x3x3 block
        maxima_in_cell_mask = tf.nn.convolution(
                tf.cast(maxima_in_cell_mask, dtype=tf.float32),
                tf.ones((3, 3, 3, 1, 1), dtype=tf.float32),
                padding='SAME',
                data_format='NCDHW')
        maxima_in_cell_mask = tf.cast(maxima_in_cell_mask,
                                      dtype=tf.bool)

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
    cell_indicator_loss = tf.losses.mean_squared_error(
        # l=1, d, h, w
        gt_cell_indicator,
        # l=1, d, h, w
        cell_indicator,
        # l=1, d, h, w
        cell_mask)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)

    # smooth transition from training parent vectors on complete cell mask to
    # only on maxima
    # https://www.wolframalpha.com/input/?i=1.0%2F(1.0+%2B+exp(0.01*(-x%2B20000)))+x%3D0+to+40000
    alpha = tf.constant(1.0)/(
        tf.constant(1.0) + tf.exp(
            tf.constant(0.01) *
            (-iteration + tf.constant(20000.0))
        )
    )

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

    tf.train.export_meta_graph(filename=setup_dir + '/' + name + '.meta')

    names = {
        'raw': raw.name,
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

    with open(setup_dir + '/' + name + '_config.json', 'w') as f:
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
    parser.add_argument('--only', type=str,
                        help="'train' or 'test', to only make one network")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    if args.only != "test":
        create_network(
            tuple(config['train_input_shape']),
            'train_net',
            config['constant_upsample'],
            config['average_vectors'],
            config['setup_dir'])
    if args.only != "train":
        create_network(
            tuple(config['predict_input_shape']),
            'test_net',
            config['constant_upsample'],
            config['average_vectors'],
            config['setup_dir'])
