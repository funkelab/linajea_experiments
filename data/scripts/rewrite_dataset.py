import daisy
import argparse
import logging
import numpy as np
import time
import pyklb
import os.path

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def get_downsample_factors(source_vs, target_vs):
    if target_vs is None:
        return None
    downsample_factor = [1 for dim in range(len(source_vs))]
    downsample_flag = False
    for dim, source in enumerate(source_vs):
        target = target_vs[dim]
        if source < target:
            assert target % source == 0,\
                ("Expected source voxel size %s to evenly divide"
                 "target %s at dim %d")\
                % (source, target, dim)
            downsample_factor[dim] = target // source
            downsample_flag = True
    return downsample_factor if downsample_flag else None


def get_upsample_factors(source_vs, target_vs):
    if target_vs is None:
        return None
    upsample_factor = [1 for dim in range(len(source_vs))]
    upsample_flag = False
    for dim, source in enumerate(source_vs):
        target = target_vs[dim]
        if source > target:
            assert source % target == 0,\
                ("Expected target voxel size %s to evenly divide"
                 "source %s at dim %d")\
                % (target, source, dim)
            upsample_factor[dim] = source // target
            upsample_flag = True
    return upsample_factor if upsample_flag else None


def resample(data, upsample_factor, downsample_factor):
    if downsample_factor:
        slices = tuple(slice(None, None, factor)
                       for factor in downsample_factor)
        tmp = data[slices]
    else:
        tmp = data
    if upsample_factor:
        for dim, factor in enumerate(upsample_factor):
            tmp = np.repeat(tmp, factor, axis=dim)
    return tmp


def resample_block(
        block,
        sources,
        group,
        attr_filenames,
        upsampling_factors,
        downsampling_factors,
        target,
        voxel_size):
    logger.debug("Resampling block %s" % block)
    start_time = time.time()
    if attr_filenames is None:
        attr_filenames = [None for dim in range(len(sources))]
    channels = []
    for source, attr_file, upsample, downsample in zip(
            sources, attr_filenames, upsampling_factors, downsampling_factors):
        ds = daisy.open_ds(source, group, attr_filename=attr_file)
        read_roi = block.read_roi.intersect(ds.roi)
        data = ds.to_ndarray(roi=read_roi)
        if upsample is None and downsample is None:
            channels.append(data)
        else:
            channels.append(resample(data, upsample, downsample))
    target_ds = daisy.open_ds(target, args.target_group, 'a')
    if len(channels) > 1:
        target_ds[block.write_roi] = np.stack(channels, axis=0)
    else:
        target_ds[block.write_roi] = channels[0]
    logger.info("Done with block %s in %s"
                % (block, time.time() - start_time))
    return


def write_klb(
        block,
        source,
        group,
        target,
        max_value,
        attr_filename=None):
    logger.debug("Inside write_klb function")
    ds = daisy.open_ds(source, group, attr_filename=attr_filename)
    logger.debug("opened ds")
    read_roi = block.read_roi.intersect(ds.roi)
    data = ds.to_ndarray(roi=read_roi)
    logger.debug("Read data")
    target_base, target_end = os.path.splitext(target)
    time = "_TM%03d" % block.read_roi.get_offset()[0]
    target_filename = target_base + time + target_end
    logger.info("Target filename: %s" % target_filename)
    max_range = np.iinfo(np.uint16).max
    logger.info("Max range: %d" % max_range)
    logger.info("Max value: %f" % max_value)
    data = (data / max_value * max_range).astype(np.uint16)
    pyklb.writefull(
            data,
            target_filename,
            pixelspacing_tczyx=tuple((1,) + ds.voxel_size))
    logger.info("Wrote file %s with header %s"
                % (target_filename, pyklb.readheader(target_filename)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "source", type=str, nargs='+', help="Path to source file")
    parser.add_argument(
            "--attr_filename", type=str, nargs='*',
            help="Filenames of attribute files for source")
    parser.add_argument(
            "--group", type=str,
            help="Group name for zarr source")
    parser.add_argument(
            "target", type=str, help="Path to output file")
    parser.add_argument(
            "--target_group", type=str, help="Group name for zarr target")
    parser.add_argument(
            "--voxel_size", nargs='*', type=int,
            help="Target voxel size after resampling")
    parser.add_argument(
            "--block_size", nargs='+',
            help="Block size to use for reading and writing")
    parser.add_argument(
            "--workers", "-w", type=int, default=10,
            help="Number of workers to use for reading and writing")
    parser.add_argument(
            "--max_val", type=float,
            help="Max value of data for klb conversion to uint16")
    parser.add_argument(
            "--frames", type=int, nargs=2, default=None,
            help="Range of frames to consider")
    args = parser.parse_args()

    logging.info("Target voxel size is %s" % str(args.voxel_size))
    target_voxel_size = args.voxel_size if args.voxel_size else None
    if args.frames is not None:
        total_roi = daisy.Roi((args.frames[0], None, None, None),
                              (args.frames[1] - args.frames[0], None, None, None))
    else:
        total_roi = None
    dtype = None
    upsampling_factors = []
    downsampling_factors = []
    for ind, source in enumerate(args.source):
        logging.info("Reading file %s" % source)
        attr_filename = None
        if args.attr_filename:
            attr_filename = args.attr_filename[ind]
            logging.info("Using attributes file %s" % attr_filename)
        group_name = args.group if args.group else None
        logging.info("Source group_name: %s" % group_name)
        source_ds = daisy.open_ds(source, group_name, attr_filename=attr_filename)
        logger.debug("Source %s has roi %s" % (source, source_ds.roi))
        logger.debug("Source %s has voxel_size %s"
                     % (source, source_ds.voxel_size))
        logger.debug("Source %s has dtype %s" % (source, source_ds.dtype))

        dtype = source_ds.dtype
        if total_roi is None:
            total_roi = source_ds.roi
        else:
            total_roi = total_roi.intersect(source_ds.roi)
        source_voxel_size = source_ds.voxel_size
        upsampling_factors.append(
                get_upsample_factors(source_voxel_size, target_voxel_size)
                if target_voxel_size else None)
        downsampling_factors.append(
                get_downsample_factors(source_voxel_size, target_voxel_size)
                if target_voxel_size else None)
    if target_voxel_size:
        total_roi = total_roi.snap_to_grid(target_voxel_size)
    else:
        target_voxel_size = source_voxel_size
    logger.debug("Total roi shape: %s" % str(total_roi.get_shape()))

    logging.info("Preparing target dataset at %s" % args.target)
    if not args.target.endswith('.klb'):
        assert args.block_size
        block_roi = daisy.Roi((0, 0, 0, 0), tuple(args.block_size))
        target_ds = daisy.prepare_ds(
                args.target,
                args.target_group,
                total_roi,
                target_voxel_size,
                dtype=source_ds.dtype,
                write_size=block_roi.get_shape(),
                num_channels=len(args.source))
        logging.info("Running blockwise with total_roi %s and block roi %s"
                     % (total_roi, block_roi))
        daisy.run_blockwise(
                total_roi,
                block_roi,
                block_roi,
                process_function=lambda b: resample_block(
                    b,
                    args.source,
                    args.group,
                    args.attr_filename,
                    upsampling_factors,
                    downsampling_factors,
                    args.target,
                    target_voxel_size),
                num_workers=args.workers,
                fit='shrink')
    else:
        klb_block_shape = (1,) + tuple(total_roi.get_shape()[1:])
        klb_block_roi = daisy.Roi((0,0,0,0), klb_block_shape)
        daisy.run_blockwise(
                total_roi,
                klb_block_roi,
                klb_block_roi,
                process_function=lambda b: write_klb(
                    b,
                    args.source[0],
                    args.group,
                    args.target,
                    args.max_val),
                num_workers=args.workers,
                fit='shrink')
