import neuroglancer as ng
import argparse
import zarr
import webbrowser
import numpy as np
from linajea import CandidateDatabase
import daisy
import logging

logger = logging.getLogger('__main__')
logging.basicConfig(level=logging.INFO)


def adjust(data):
    if data.max() <= 1.0:
        data = data*254
    else:
        np.clip(data, 0, 254, data)
    data = data.astype(np.uint8)
    print("Dtype of frame data: %s " % data.dtype)
    print("MAx value in frame data: %f" % data.max())
    print("Mean value in frame data: %f" % data.mean())
    print("Shape of frame data: %s" % str(data.shape))
    return data


def add_layer(context, data, name, voxel_size, roi):
    print("Layer name %s" % name)
    print("voxel size: %s" % str(voxel_size[-1:-4:-1]))
    print("Offset: %s" % str(roi.get_offset()[-1:-4:-1]))
    visible = True
    layer = ng.LocalVolume(
        data=adjust(data),
        volume_type='image',
        voxel_size=voxel_size[-1:-4:-1],
        offset=roi.get_offset()[-1:-4:-1])

    context.layers.append(
            name=name,
            layer=layer,
            visible=visible)


def add_annotations(context, points, name, color='#ff00ff'):
    visible = True
    annotations = [
            ng.EllipsoidAnnotation(
                center=tuple(point),
                radii=(10, 10, 10),
                id=_id,
                segments=None)
            for _id, point in points.items()]
    print("%d annotations" % len(annotations))

    layer = ng.AnnotationLayer(
            filter_by_segmentation=False,
            voxel_size=(1, 1, 1),
            annotations=annotations,
            annotation_color=color)

    context.layers.append(
            name=name,
            layer=layer,
            visible=visible)


def get_points_in_t(tracksfile, roi):
    time = roi.get_offset()[0]
    points = {}
    with open(tracksfile, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            t = int(tokens[0])
            if t == time:
                z = float(tokens[1])
                y = float(tokens[2])
                x = float(tokens[3])
                _id = int(tokens[4])
                points[_id] = [x, y, z]
                logger.debug("Point %d at %s" % (_id, str([x, y, z])))
    return points


def get_results(database, roi, key=None, node_score_threshold=None):
    # TODO: add edges as lines going backward in time
    host = "localhost"
    db = CandidateDatabase(database, host, 'r', parameters_id=key)
    if key:
        selected_graph = db.get_selected_graph(roi)
        points_list = []
        for node_id, data in selected_graph.nodes(data=True):
            point = data
            point['id'] = node_id
            points_list.append(point)
    else:
        points_list = db.read_nodes(roi)
    points_by_time = {}
    for point in points_list:
        if node_score_threshold:
            score = point['score']
            if score < node_score_threshold:
                continue

        _id = point['id']
        z = point['z']
        y = point['y']
        x = point['x']
        t = point['t']
        if t not in points_by_time:
            points_by_time[t] = {}
        points_by_time[t][_id] = [x, y, z]
        logger.debug("Point %d at %s" % (_id, str([x, y, z])))
    return points_by_time


def visualize_data(
        datafile,
        dataset,
        time,
        channels=False,
        annotations=None,
        results=None,
        node_score_threshold=None,
        key=None,
        mask=False):

    ng.set_server_bind_address(bind_address='0.0.0.0', bind_port=0)
    viewer = ng.Viewer()
    z = zarr.open(datafile, 'r')
    data = z[dataset]
    print("Shape of all data: %s" % str(data.shape))

    print("Attributes: %s" % str(data.attrs.asdict()))
    voxel_size = daisy.Coordinate(data.attrs['resolution'])
    if datafile.endswith('n5'):
        voxel_size = voxel_size[::-1]
    print("Voxel size: %s" % str(voxel_size))
    if channels:
        voxel_shape = [s * v for s, v in zip(data.shape[1:], voxel_size)]
    else:
        voxel_shape = [s*v for s, v in zip(data.shape, voxel_size)]
    if len(time) == 1:
        time_start = time[0]
        time_size = 1
    elif len(time) == 2:
        time_start, time_end = time
        time_size = time_end - time_start
    else:
        raise ValueError("Must provide one time or range of times")
    start = (time_start, 0, 0, 0)
    shape = (time_size, ) + tuple(voxel_shape[1:])
    roi = daisy.Roi(start, shape)
    print("Offset: %s" % str(roi.get_offset()))
    offset_voxels = roi.get_offset()/voxel_size
    shape_voxels = roi.get_shape()/voxel_size
    print("Shape voxels: %s" % str(shape_voxels))
    sl = tuple(slice(o, o+s) for o, s in zip(offset_voxels, shape_voxels))
    if channels:
        sl = (slice(None),) + sl
    print("Slice: %s" % str(sl))
    data = data[sl]

    print("Shape of data: %s" % str(data.shape))

    if mask:
        mask_data = z['volumes/mask']
        print("Shape of whole mask: %s" % str(mask_data.shape))
        mask_data = mask_data[sl]
        print("Shape of mask: %s" % str(mask_data.shape))

    # frame = frame.swapaxes(-1, -3)

    if annotations:
        points = get_points_in_t(annotations, roi)
    if results:
        print("Getting results from db %s" % results)
        result_points = get_results(
                results, roi, key=key,
                node_score_threshold=node_score_threshold)
    with viewer.txn() as s:
        if channels:
            for frame in range(data.shape[1]):
                for c in range(data.shape[0]):
                    add_layer(
                            s,
                            data[c][frame],
                            dataset + "_" + str(c),
                            voxel_size,
                            roi)
        else:
            for frame in range(data.shape[0]):
                add_layer(
                        s,
                        data[frame],
                        dataset + str(frame),
                        voxel_size,
                        roi)
        if mask:
            if channels:
                for frame in range(mask_data.shape[1]):
                    for c in range(mask_data.shape[0]):
                        add_layer(
                                s,
                                mask_data[c][frame],
                                'mask_' + str(c),
                                voxel_size,
                                roi)
            else:
                for frame in range(mask_data.shape[0]):
                    add_layer(
                            s,
                            mask_data[frame],
                            'mask' + str(frame),
                            voxel_size,
                            roi)
        if annotations:
            add_annotations(s, points, 'cells', color='#00FF00')
        if results:
            print("Adding results to viewer")
            for t, points in result_points.items():
                add_annotations(
                        s, points, 'cell-cands-%d' % t, color='#FFFF00')

    return viewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help="path to data")
    parser.add_argument('-d', '--dataset', help="dataset to visualize")
    parser.add_argument(
            '-t', '--time', type=int, nargs='+',
            help="time points to visualize (slice)")
    parser.add_argument(
            '-c', '--channels', action='store_true',
            help="flag indicating that there are multiple channels")
    parser.add_argument(
            '-a',
            '--annotations',
            help='path to tracks file to visualize points from')
    parser.add_argument(
            '-r', '--results', help='mongo db with results')
    parser.add_argument(
            '--node_threshold', type=float,
            help='score threshold to filter node candidates')
    parser.add_argument(
            '-k', '--key', help='selected_key to use when filtering results')
    parser.add_argument(
            '-m', '--mask', help='show mask at volumes/mask', action='store_true')
    args = parser.parse_args()

    viewer = visualize_data(
        args.file,
        args.dataset,
        args.time,
        args.channels,
        args.annotations,
        args.results,
        args.node_threshold,
        args.key,
        args.mask)
    url = str(viewer)
    print(url)
    webbrowser.open_new(url)

    print("Press ENTER to quit")
    input()
