import argparse
import json
import logging
import os
import tensorflow.compat.v1 as tf
import toml

from vgg import vgg

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
        num_classes,
        is_training,
        lr,
        num_fmaps=12,
        kernel_sizes=None,
        downsample_factors=None,
        fmap_inc_factors=None,
        padding='same',
        activation='relu',
        batch_normalize=True,
        fc_size=512,
        global_pool=True,
        setup_dir=None,
        batch_size=None):

    tf.reset_default_graph()

    if not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    # create a placeholder for the 3D raw input tensor
    raw_batched = tf.placeholder(tf.float32,
                                 shape=(batch_size,) + input_shape,
                                 name="raw")

    # create the VGG network
    logits, sums = vgg(
        raw_batched,
        num_classes=num_classes,
        is_training=is_training,
        num_fmaps=num_fmaps,
        kernel_sizes=kernel_sizes,
        downsample_factors=downsample_factors,
        fmap_inc_factors=fmap_inc_factors,
        padding=padding,
        activation=activation,
        batch_normalize=batch_normalize,
        fc_size=fc_size,
        global_pool=global_pool)
    print(logits)

    pred_labels = tf.argmax(input=logits, axis=1, name="classes")
    print(pred_labels)
    pred_probs = tf.nn.softmax(logits, name="softmax_tensor")
    print(pred_probs)

    gt_labels = tf.placeholder(tf.int32, shape=[None], name="gt_labels")
    print(gt_labels)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels,
                                                  logits=logits)

    sums.append(tf.summary.scalar('loss_sum', loss))
    summaries = tf.summary.merge(sums, name="summaries")

    # optimizer
    learning_rate = tf.placeholder_with_default(lr, shape=(),
                                                name="learning-rate")
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)
    optimizer = opt.minimize(loss=loss,
                             global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([optimizer, update_ops])

    os.makedirs(setup_dir, exist_ok=True)

    fn = os.path.join(setup_dir, 'vgg')
    if is_training:
        fn += '_train'
    tf.train.export_meta_graph(filename=fn + '.meta')

    names = {
        'raw': raw_batched.name,
        'gt_labels': gt_labels.name,
        'pred_labels': pred_labels.name,
        'pred_probs': pred_probs.name,
        'loss': loss.name,
        'optimizer': train_op.name,
        'summaries': summaries.name
    }

    with open(fn + '_names.json', 'w') as f:
        json.dump(names, f)

    config = {
        'input_shape': input_shape,
    }

    with open(fn + '_config.json', 'w') as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--only', type=str,
                        help="'train' or 'test', to only make one network")
    args = parser.parse_args()
    with open(args.config) as f:
        config = toml.load(f)
    if args.only != "test":
        create_network(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            is_training=True,
            lr=config['lr'],
            num_fmaps=config['num_fmaps'],
            kernel_sizes=config['kernel_sizes'],
            downsample_factors=config['downsample_factors'],
            fmap_inc_factors=config['fmap_inc_factors'],
            setup_dir=config['setup_dir'],
            batch_size=config['batch_size'])
    if args.only != "train":
        create_network(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            is_training=False,
            lr=config['lr'],
            num_fmaps=config['num_fmaps'],
            kernel_sizes=config['kernel_sizes'],
            downsample_factors=config['downsample_factors'],
            fmap_inc_factors=config['fmap_inc_factors'],
            setup_dir=config['setup_dir'],
            batch_size=1)
