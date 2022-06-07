import pyklb
import argparse
import os
import logging
import numpy as np
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transpose_y_z(klb_file, new_klb_file, transpose_blocksize=False):
    header = pyklb.readheader(klb_file)
    logging.debug("KLB header: %s" % header)
    array = pyklb.readfull(klb_file)
    logging.debug("Type of array: %s" % type(array))
    logging.debug("Shape of array: %s" % str(array.shape))
    #                                 y  z  x
    transposed = np.ascontiguousarray(np.transpose(array, (1, 0, 2)))
    pixelspacing = header['pixelspacing_tczyx']
    pixelspacing[2], pixelspacing[3] = pixelspacing[3], pixelspacing[2]
    pixelspacing = tuple(pixelspacing)
    logger.debug("Transposed pixelspacing: %s" % str(pixelspacing))
    blocksize = header['blocksize_tczyx']
    if transpose_blocksize:
        blocksize[2], blocksize[3] = blocksize[3], blocksize[2]
    blocksize = blocksize[::-1]
    
    logger.debug("Using blocksize xyzct: %s" % blocksize)

    pyklb.writefull(
            transposed,
            new_klb_file,
            pixelspacing_tczyx=pixelspacing,
            blocksize_xyzct=blocksize)
    logger.debug("Wrote file %s with header %s"
                 % (new_klb_file, pyklb.readheader(new_klb_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'source_dir',
            type=str,
            help='Path to directory containing klb files')
    parser.add_argument(
            'target_dir',
            type=str,
            help='Path to directory to put transposed files in')
    parser.add_argument(
            '--limit_to',
            type=str,
            default=None,
            help='Filename format (unix style wildcards). '
                 'Will only transpose files with this format.')
    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    filename = args.limit_to

    assert os.path.isdir(source_dir)
    assert os.path.isdir(target_dir)

    if filename:
        pathname = os.path.join(source_dir, filename)
        source_files = [os.path.basename(p) for p in glob.glob(pathname)]
    else:
        source_files = [f for f in os.listdir(source_dir) if f.endswith('klb')]

    logger.debug(source_files)

    for source_file in source_files:
        root, ext = os.path.splitext(source_file)
        assert ext == '.klb', "Extension is not klb"
        target_file = root + '.transposed' + ext

        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, target_file)
        logging.info("Transposing y and z in %s, writing to %s"
                     % (source_path, target_path))
        # Looks like blocksize was not transposed originally
        # for the zebrafish data ->
        # using the same one should be correct
        transpose_y_z(source_path, target_path, transpose_blocksize=False)
