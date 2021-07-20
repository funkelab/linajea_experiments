from scipy.ndimage.filters import gaussian_filter
import numpy as np
import argparse
import daisy
import logging
from linajea import load_config
import json
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def create_mask_for_frame(
        roi_offset,
        roi_shape,
        input_file,
        input_dataset,
        output_file,
        sigma,
        min_percentile=1,
        downsample=100,
        min_mean_distance=.4,
        max_coverage=1.0,
        attrs=None,
        channels=1,
        write_params=False):
    dataset = daisy.open_ds(
            input_file, input_dataset,
            attr_filename=attrs,
            mode='a')
    output_dataset = daisy.open_ds(
            output_file, 'volumes/mask', 'a')
    read_roi = daisy.Roi(roi_offset, roi_shape)
    frame = dataset[read_roi]
    logger.debug("Loading image at roi %s" % str(read_roi))
    image = np.squeeze(frame.to_ndarray())
    logger.info("Shape of image: %s", str(image.shape))
    if channels > 1:
        masks = []
        params = []
        for c in range(channels):
            m, p = generate_mask(
                    image[c],
                    sigma,
                    min_percentile=min_percentile,
                    downsample=downsample,
                    min_mean_distance=min_mean_distance,
                    max_coverage=max_coverage)
            masks.append(m)
            params.append(p)
        mask = np.array(masks)
        mask = np.expand_dims(mask, axis=1)
    else:
        mask, params = generate_mask(
                image,
                sigma,
                min_percentile=min_percentile,
                downsample=downsample,
                min_mean_distance=min_mean_distance,
                max_coverage=max_coverage)
        mask = np.expand_dims(mask, axis=0)
    logger.info("Shape of mask: %s" % str(mask.shape))
    output_dataset[read_roi] = mask

    if write_params:
        time = roi_offset[0]
        with open(os.path.join(
                output,
                'volumes',
                'mask',
                str(time) + '_params.json'), 'w') as f:
            json.dump(params, f)


def generate_mask(
        array,
        sigma,
        min_percentile=1,
        downsample=100,
        min_mean_distance=.4,
        max_coverage=1.0):
    ''' Generates a mask around the section of an image that contains data.
    First does gaussian smoothing, then chooses a threshold based on a set
    distance between the min percentile and the mean value.
    '''
    logger.debug("Smoothing")
    smoothed = gaussian_filter(array, sigma=sigma, mode='nearest')
    logger.debug("Calculating min percentile")
    lower = np.percentile(smoothed.ravel()[::downsample], min_percentile)
    logger.info("Min percentile value is %f" % lower)
    logger.debug("Computing mean")
    mean = smoothed.mean()
    logger.info("Mean value is %f" % mean)
    coverage = 1.1
    while coverage > max_coverage:
        threshold = lower + (min_mean_distance * (mean - lower))
        logger.info("Threshold value is %f" % threshold)
        logger.debug("Computing mask")
        mask = (smoothed > threshold).astype(np.bool)
        coverage = np.count_nonzero(mask) / mask.size
        logger.info("Mask covers %.2f of image", coverage)
        if coverage > max_coverage:
            min_mean_distance += 0.01
            logger.info("Coverage %.2f is greater than max allowed %.2f."
                        " Retrying with min_mean_distance %f",
                        coverage, max_coverage, min_mean_distance)
    logger.debug("Done generating mask")
    params = {
            'sigma': sigma,
            'min_percentile': min_percentile,
            'downsample': downsample,
            'min_mean_distance': min_mean_distance,
            'threshold': threshold,
            'max_coverage': max_coverage
            }

    return mask, params


def save_mask_for_image(
        input_file,
        times,
        output,
        sigma,
        dataset_name=None,
        attrs=None,
        min_percentile=1,
        downsample=100,
        min_mean_distance=0.4,
        max_coverage=1.0,
        parallel=False,
        num_workers=5,
        channels=1,
        write_params=False):
    dataset = daisy.open_ds(
            input_file, dataset_name,
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
            num_channels=channels)
    if not parallel:
        for time in range(*times):
            logger.info("Processing time %d" % time)
            frame_offset = (time,) + total_roi.get_offset()[1:]
            create_mask_for_frame(
                    frame_offset,
                    frame_shape,
                    input_file,
                    dataset_name,
                    output,
                    sigma,
                    min_percentile=min_percentile,
                    downsample=downsample,
                    min_mean_distance=min_mean_distance,
                    max_coverage=max_coverage,
                    attrs=attrs,
                    channels=channels)
    else:
        total_roi = daisy.Roi(
                (times[0],) + total_roi.get_offset()[1:],
                (times[1] - times[0],) + total_roi.get_shape()[1:])
        daisy.run_blockwise(
                total_roi,
                frame_roi,
                frame_roi,
                process_function=lambda b: create_mask_for_frame(
                        b.read_roi.get_offset(),
                        b.read_roi.get_shape(),
                        input_file,
                        dataset_name,
                        output,
                        sigma,
                        min_percentile=min_percentile,
                        downsample=downsample,
                        min_mean_distance=min_mean_distance,
                        max_coverage=max_coverage,
                        attrs=attrs,
                        channels=channels),
                read_write_conflict=False,
                num_workers=num_workers,
                max_retries=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument(
            '-p', '--parallel', action='store_true')
    parser.add_argument(
            '-c', '--channels', type=int, default=1)
    parser.add_argument(
            '-wp', '--write-params', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config_file)
    input_file = config['input_file']
    output = config['output_file']
    times = config['times']
    dataset = config.get('dataset', None)
    attrs = config.get('attributes', None)
    sigma = config.get('sigma', [1, 2, 2])
    min_percentile = config.get('min_percentile', 1)
    downsample = config.get('downsample', 100)
    min_mean_distance = config.get('min_mean_distance', 0.4)
    max_coverage = config.get('max_coverage', 1.0)
    num_workers = config.get('num_workers', 5)
    save_mask_for_image(
        input_file,
        times,
        output,
        sigma,
        dataset_name=dataset,
        attrs=attrs,
        min_percentile=min_percentile,
        downsample=downsample,
        min_mean_distance=min_mean_distance,
        max_coverage=max_coverage,
        parallel=args.parallel,
        num_workers=num_workers,
        channels=args.channels,
        write_params=args.write_params)
