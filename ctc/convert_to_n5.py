import zarr
import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import os
import argparse
import logging

logger = logging.getLogger(__name__)


def convert_to_n5(directory, outfile):
    assert os.path.exists(directory)
    files = sorted(glob.glob(os.path.join(directory, '*.tif')))
    assert len(files) > 0, f"No tif files in {directory}"
    logger.debug(files)

    raw = np.array([imread(f) for f in tqdm(files)])
    logger.debug(raw.shape)
    logger.debug(raw.dtype)
    logger.debug(raw.min(), raw.max())

    f = zarr.open(outfile, 'a')
    f['raw'] = raw
    # x, y, z, t for N5
    f['raw'].attrs['resolution'] = [1, 1, 5, 1]


if __name__ == '__main__':
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Path to training data directory", default=None)
    parser.add_argument("--test", help="Path to test data directory", default=None)
    parser.add_argument("outdir", help="Directory to write output n5 files to")
    parser.add_argument('-o', '--only', default=None)
    args = parser.parse_args()
    if args.train is not None:
        if not args.only == '2':
            convert_to_n5(os.path.join(args.train, '01'),
                          os.path.join(args.outdir, 'train_01.n5'))
        if not args.only == '1':
            convert_to_n5(os.path.join(args.train, '02'),
                          os.path.join(args.outdir, 'train_02.n5'))
    if args.test is not None:
        if not args.only == '2':
            convert_to_n5(os.path.join(args.test, '01'),
                          os.path.join(args.outdir, 'test_01.n5'))
        if not args.only == '1':
            convert_to_n5(os.path.join(args.test, '02'),
                          os.path.join(args.outdir, 'test_02.n5'))
