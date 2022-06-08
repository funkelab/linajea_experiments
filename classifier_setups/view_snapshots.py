from funlib.show.neuroglancer import add_layer
import argparse
import daisy
import neuroglancer
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file',
    '-f',
    type=str,
    help="The path to the container to show")

args = parser.parse_args()

neuroglancer.set_server_bind_address('0.0.0.0')
viewer = neuroglancer.Viewer()

shader = '''
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(1)))
        );
}
'''

f = args.file
for ds in ['volumes/raw']:

    arrays = []

    print("Adding %s, %s" % (f, ds))
    batch = daisy.open_ds(f, ds)

    with viewer.txn() as s:
        for i in range(len(batch.data)):
            a = daisy.Array(
                batch.data[i]*100.0,
                roi=daisy.Roi(
                    batch.roi.get_begin()[1:],
                    batch.roi.get_shape()[1:]),
                voxel_size=daisy.Coordinate(batch.voxel_size[1:]))

            add_layer(s, a, ds + f'_{i}', shader=shader)

print("pred_probs:", h5py.File(f, 'r')['/pred_probs'][:])
# print("pred_labels:", h5py.File(f, 'r')['/pred_labels'][:])
print("gt_labels:", h5py.File(f, 'r')['/gt_labels'][:])

url = str(viewer)
print(url)

print("Press ENTER to quit")
input()
