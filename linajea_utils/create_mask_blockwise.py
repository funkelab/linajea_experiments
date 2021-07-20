import argparse
import daisy
import logging
import time
from linajea import load_config
from funlib.run import run

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def mask_worker(block, config_file):
    config = load_config(config_file)
    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    image_path = '/nrs/funke/singularity/'
    singularity_image = config['singularity_image']
    image = image_path + singularity_image + '.img'
    logger.debug("Using singularity image %s" % image)
    offset_string = ' '.join(map(str, block.read_roi.get_offset()))
    shape_string = ' '.join(map(str, block.read_roi.get_shape()))
    command = 'python -u %s --config %s --block_roi_offset %s --block_roi_shape %s' % (
                'create_mask.py',
                config_file,
                offset_string,
                shape_string)
    cmd = run(
            command=command,
            num_cpus=5,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            )
    cmd = command.split(' ')
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    daisy.call(
        cmd,
        log_out='logs/create_mask_%d_%d.out' % (worker_time, worker_id),
        log_err='logs/create_mask_%d_%d.err' % (worker_time, worker_id))

    logger.info("Mask  worker finished")


def save_mask_for_image(config_file):
    config = load_config(config_file)
    input_file = config['input_file']
    output = config['output_file']
    times = config['times']
    dataset = config.get('dataset', None)
    attrs = config.get('attributes', None)

    dataset = daisy.open_ds(
            input_file, dataset,
            attr_filename=attrs)
    total_roi = dataset.roi
    frame_shape = (1,) + total_roi.get_shape()[1:]
    frame_roi = daisy.Roi(total_roi.get_offset(), frame_shape)
    daisy.prepare_ds(
            output,
            'volumes/mask',
            total_roi,
            dataset.voxel_size,
            bool,
            write_size=frame_shape,
            nested=True)

    total_roi = daisy.Roi(
            (times[0],) + total_roi.get_offset()[1:],
            (times[1] - times[0],) + total_roi.get_shape()[1:])
    daisy.run_blockwise(
            total_roi,
            frame_roi,
            frame_roi,
            process_function=lambda b: mask_worker(
                    b,
                    config_file),
            read_write_conflict=False,
            num_workers=5,
            max_retries=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--config_file', help="Path to config file")
    args = parser.parse_args()
    save_mask_for_image(
        args.config_file)
