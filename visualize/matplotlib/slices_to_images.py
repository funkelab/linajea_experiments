import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import zarr
import pyklb
import numpy as np
import logging
import time
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

colorscheme = [
    '#000000',
    '#808080',
    '#556b2f',
    '#8b4513',
    '#8b0000',
    '#808000',
    '#483d8b',
    '#008000',
    '#008080',
    '#4682b4',
    '#9acd32',
    '#00008b',
    '#32cd32',
    '#daa520',
    '#8fbc8f',
    '#800080',
    '#b03060',
    '#d2b48c',
    '#ff0000',
    '#00ced1',
    '#ff8c00',
    '#ffff00',
    '#7fff00',
    '#dc143c',
    '#f4a460',
    '#0000ff',
    '#a020f0',
    '#ff00ff',
    '#1e90ff',
    '#f0e68c',
    '#fa8072',
    '#dda0dd',
    '#90ee90',
    '#87ceeb',
    '#ff1493',
    '#7b68ee',
    '#ee82ee',
    '#7fffd4',
    '#fffaf0',
    '#ffc0cb',
    ]


def get_color_cmap(
        color=[255./255., 226./255., 100./255., 1.],
        color_point=200):
    # set color_point = 256 to end at the given color
    # otherwise it fades to white in the remaining segment
    intensities = np.linspace(0, 1, color_point)
    color_cmap = [[0., 0., 0., 1.]]*256
    for i in range(color_point):
        color_cmap[i] = [
                intensities[i] * color[j]
                for j in range(3)
                ]

    steps = 256 - color_point
    for i in range(color_point, 256):
        curr_step = i - color_point
        inc_per_step = [(1. - color[i])/steps
                        for i in range(3)]
        color_cmap[i] = [
                color[j] + (curr_step + 1) * inc_per_step[j]
                for j in range(3)
                ]
    return ListedColormap(color_cmap)


def get_points(points, t, z, y_slice, x_slice, radius=10, zscale=5):
    pointset = []
    with open(points) as f:
        for line in f.readlines():

            # node_id, t, z, y, x, node_score, edge_score, pid, track_id
            tokens = line.strip().split(',')
            tokens.pop(6)
            tokens.pop(5)
            tokens = list(map(int, tokens))
            node_id, node_t, node_z, node_y, node_x, pid, track_id = tokens
            if node_t == t:
                if node_z > z*zscale - radius and node_z < z*zscale + radius:
                    if node_y >= y_slice.start and node_y < y_slice.stop:
                        if node_x >= x_slice.start and node_x < x_slice.stop:
                            pointset.append(tokens)
    return pointset


def save_image_from_zarr(
        zarr_name,
        group,
        output_name,
        output_dir,
        t_range,
        z_range,
        y_slice,
        x_slice,
        channels=1,
        ndim=4,
        yellow=False,
        norm='minmax',
        _range=None,
        points=None):
    if yellow:
        yellow = get_color_cmap()

    zarr_file = zarr.open(zarr_name, 'r')
    volume = zarr_file[group]
    shape = volume.shape
    logging.debug("Volume shape: %s" % str(shape))
    if output_name is None:
        out_name = "%d_%d_" + group + ".png"
    else:
        out_name = "%d_%d_" + output_name
    out_name = os.path.join(output_dir, out_name)
    assert t_range is not None
    ys = slice(*y_slice) if y_slice is not None else None
    xs = slice(*x_slice) if x_slice is not None else None
    zs = range(*z_range) if z_range is not None else None
    ts = range(*t_range) if t_range is not None else None
    for t in ts:
        for z in zs:
            save_image(volume, t, z, ys, xs, out_name, channels=channels,
                       yellow=yellow, norm=norm, _range=_range, points=points)


def save_image_from_klb(
        klb_name,
        group,
        output_name,
        output_dir,
        t_range,
        z_range,
        y_slice,
        x_slice,
        channels=1,
        ndim=4,
        yellow=False,
        norm='minmax',
        _range=None,
        points=None):
    header = pyklb.readheader(klb_name % 0)
    logger.debug("Header: %s: " % str(header))
    if output_name is None:
        out_name = "%d_%d_" + group + ".png"
    else:
        out_name = "%d_%d_" + output_name
    out_name = os.path.join(output_dir, out_name)
    assert t_range is not None
    ys = slice(*y_slice) if y_slice is not None else None
    xs = slice(*x_slice) if x_slice is not None else None
    zs = range(*z_range) if z_range is not None else None
    ts = range(*t_range) if t_range is not None else None
    for t in ts:
        for z in zs:
            start_time = time.time()
            klb = pyklb.readfull(klb_name % t)
            secs = time.time() - start_time
            logging.debug("Took %d seconds (%d minutes) to read klb"
                          % (secs, secs // 60))
            shape = klb.shape
            logger.debug("Shape: %s" % str(shape))
            save_image(
                    klb, t, z, ys, xs,
                    out_name, channels=channels,
                    yellow=yellow, norm=norm, _range=_range, points=points)


def save_image(
        zvol,
        t,
        z,
        y_slice,
        x_slice,
        output_name,
        channels=1,
        yellow=False,
        norm='minmax',
        _range=None,
        points=None):
    start_time = time.time()
    if channels == 1:
        arr = zvol[t, z, y_slice, x_slice] if y_slice and x_slice\
            else zvol[t, z]
    else:
        arr = zvol[:, t, z, y_slice, x_slice] if y_slice and x_slice\
            else zvol[:, t, z]
        arr = np.moveaxis(arr, 0, -1)
    arr = arr.astype(np.float32)
    logger.debug("xyslice has type %s" % str(arr.dtype))
    # normalize to [0, 1]
    if norm == 'range':
        if not _range:
            raise ValueError("No min and max provided"
                             " but range normalization selected")
        mn, mx = _range
        logger.debug("normalizing range %d to %d" % (mn, mx))
        arr -= mn
        arr /= (mx - mn)
        logger.debug("Min value mid normalization: %f" % np.min(arr))
        logger.debug("Max value mid normalization: %f" % np.max(arr))
        arr = np.clip(arr, 0., 1.)
        logger.debug("Min value after normalization: %f" % np.min(arr))
        logger.debug("Max value after normalization: %f" % np.max(arr))
    elif norm == 'minmax':
        logger.debug("Norm is minmax: %f %f" % (np.min(arr), np.max(arr)))
        arr -= np.min(arr)
        arr /= np.max(arr)
    else:
        raise ValueError("Norm options are minmax and range")

    cmap = yellow if yellow else 'gray'

    # get required figure size in inches (reversed row/column order)
    dpi=100
    print(arr.shape)
    inches = arr.shape[1]/dpi, arr.shape[0]/dpi
    print(inches)

    # make figure with that size and a single axes
    fig, ax = plt.subplots(figsize=inches, dpi=dpi)

    # move axes to span entire figure area
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.imshow(arr, cmap=cmap,
              vmin=0., vmax=1.)

    # Get points if necessary
    if points:
        radius = 8
        zscale = 5
        pointset = get_points(
                points, t, z, y_slice, x_slice, radius=radius)
        nppoints = np.array(pointset)
        logger.debug("Max y: %f" % np.max(nppoints[:, 3]))
        logger.debug("Max y: %f" % np.max(nppoints[:, 4]))

        for point in pointset:
            node_z, node_y, node_x = point[2:5]
            track_id = point[6]
            y_offset = (node_y - y_slice.start) / y_slice.step
            x_offset = (node_x - x_slice.start) / x_slice.step
            zdiff = abs(node_z - z * zscale)
            normzdiff = zdiff // zscale
            node_rad = radius / (2 + normzdiff)
            color = colorscheme[track_id % len(colorscheme)]
            ax.add_patch(Circle((x_offset, y_offset), node_rad, color=color))

    logger.debug("Saving image at time %d and z %d" % (t, z))
    extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    plt.savefig(output_name % (t, z))# , bbox_inches='tight', pad_inches=0)
    secs = time.time() - start_time
    logger.debug("Took %d seconds (%d minutes)" % (secs, secs // 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', type=str)
    parser.add_argument('--klb', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('-t', type=int, nargs='+', default=None)
    parser.add_argument('-z', type=int, nargs='+', default=None)
    parser.add_argument('-y', type=int, nargs='+', default=None)
    parser.add_argument('-x', type=int, nargs='+', default=None)
    parser.add_argument('--out', type=str)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('-c', type=int, default=1)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--norm', type=str, default='minmax',
                        help="minmax or range")
    parser.add_argument('-r', '--range', type=float, nargs=2,
                        default=[-5, 5],
                        help="min and max values for normalization")
    parser.add_argument('-p', '--points', help="Path to tracks file")
    args = parser.parse_args()
    logger.debug("Normalization: %s" % args.norm)
    if args.zarr:
        save_image_from_zarr(
                args.zarr,
                args.group,
                args.out,
                args.outdir,
                args.t,
                args.z,
                args.y,
                args.x,
                channels=args.c,
                yellow=args.color,
                norm=args.norm,
                _range=args.range,
                points=args.points)
    elif args.klb:
        save_image_from_klb(
                args.klb,
                args.group,
                args.out,
                args.outdir,
                args.t,
                args.z,
                args.y,
                args.x,
                channels=args.c,
                yellow=args.color,
                norm=args.norm,
                _range=args.range,
                points=args.points)
