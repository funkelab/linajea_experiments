filenames = [
    'original/DA_Extended.shifted.TrackingMatrixGT.csv',
    'original/TB_Extended_corrected.TrackingMatrixGT.csv',
    'original/VE_Extended.shifted.TrackingMatrixGT.csv',
    'original/140521_CM.shifted.trackingMatrixGT.csv',
    # 'original/140521_TB.shifted.trackingMatrixGT.csv',
    # 'original/140521_DorsalAorta.shifted.trackingMatrixGT.csv',
    # 'original/140521_VE.shifted.trackingMatrixGT.csv',
    'original/140521_Notochord.shifted.trackingMatrixGT.csv',
]

cell_ids = set()
track_ids = set()

cell_id_offset = 0

with open('extended_tracks_uncorrected.txt', 'w') as out:

    for filename in filenames:

        cell_id_to_orig_id = {}
        cells = {}
        edges = []

        print("Parsing %s..." % filename)
        if filename.startswith('original/140521'):
            header = True
        else:
            header = False
        file_track_ids = set()

        for line in open(filename, 'r', encoding='utf-8-sig'):
            if header:
                header = False
                continue

            tokens = line.split(',')
            orig_cell_id, cell_type, x, y, z, r, orig_parent_id, t, confidence, track_id = (
                int(tokens[0]),
                int(tokens[1]),
                float(tokens[2]),
                float(tokens[3]),
                float(tokens[4]),
                float(tokens[5]),
                int(tokens[6]),
                int(tokens[7]),
                float(tokens[8]),
                int(tokens[9]))

            cell_id = orig_cell_id + cell_id_offset
            cell_id_to_orig_id[cell_id] = orig_cell_id
            if cell_id == 232143:
                print("FOUND CELL WITH ID 232143: %s" % line)
            if orig_parent_id != -1:
                parent_id = orig_parent_id + cell_id_offset
                cell_id_to_orig_id[parent_id] = orig_parent_id
            else:
                parent_id = -1

            # leave parent_id undefined for now...
            cells[cell_id] = [t, z, y, x, None, track_id]

            # ...just store list of edges instead
            if parent_id != -1:
                edges.append((min(cell_id, parent_id),
                              max(cell_id, parent_id)))

            assert cell_id not in cell_ids, (
                "Cell IDs not unique. offending ID: %d (%d original)"
                % (cell_id, cell_id_to_orig_id[cell_id]))
            cell_ids.add(cell_id)
            file_track_ids.add(track_id)

        for (u, v) in edges:

            assert u in cells, ("Referenced cell %d not in file"
                                % cell_id_to_orig_id[u])
            assert v in cells, ("Referenced cell %d not in file"
                                % cell_id_to_orig_id[v])

            frame_u = cells[u][0]
            frame_v = cells[v][0]

            # let u be the earlier cell
            if frame_u > frame_v:
                (u, v) = (v, u)
            elif frame_u == frame_v:
                raise RuntimeError(
                    "edge between %d and %d within frame %d"
                    % (u, v, frame_u))

            # set parent of v to u
            if cells[v][4] is not None:
                print(
                    "track does not form a tree in time, cell %d has"
                    " (at least) parents %d and %d" % (
                        cell_id_to_orig_id[v],
                        cell_id_to_orig_id[cells[v][4]],
                        cell_id_to_orig_id[u]))
            cells[v][4] = u

        for cell_id, cell in cells.items():

            t, z, y, x, parent_id, track_id = cell
            if parent_id is None:
                parent_id = -1
            out.write(
                "%f\t%f\t%f\t%f\t%d\t%d\t%d\n" %
                (t, z, y, x, cell_id, parent_id, track_id))

        cell_id_offset = max(cell_ids) + 1

        for track_id in file_track_ids:
            assert track_id not in track_ids, (
                "Track IDs not unique across files (offending ID: %d)"
                % track_id)
