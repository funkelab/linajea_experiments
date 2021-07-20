from scipy.io import loadmat

mat = loadmat('original/NervousSystem.mat')
tracks = mat['trackingNeuroblastCurated']

print(tracks.dtype)

with open('tracks.txt', 'w') as f:

    for cell in tracks:

        # for some reason, annotations are stored in (y, x, z)...
        _id, y, x, z, parent_id, time = cell
        track_id = -1

        assert _id <= 9007199254740993, (
            "Cell IDs exceed 9007199254740993, possible loss of precision "
            "(cell IDs in mat-file are stored as float64)")
        assert round(_id) == _id
        assert round(parent_id) == parent_id
        assert round(track_id) == track_id

        f.write(
            "%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (
                round(time),
                round(z*5),
                round(y),
                round(x),
                int(_id),
                int(parent_id),
                int(track_id)))
