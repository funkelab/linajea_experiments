from linajea import CandidateDatabase, load_config
from daisy import Roi, Coordinate
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_tracks_to_mongo(tracks_file, db_host, db_name,
                        has_div_state=False, has_radius=False,
                        has_name=False, frames=None, points=False):

    db = CandidateDatabase(db_name, db_host, mode='w')
    nodes = {}
    min_dims = None
    max_dims = None

    with open(tracks_file, 'r') as fl:
        ln = fl.readline()
        if "parent" in ln and "track" in ln:
            header = ln.split()
            print("using header {}".format(header))
        else:
            header = None
            fl.seek(0)

        for line in fl:
            tokens = line.split()
            cell = {}
            if header is not None:
                for key, val in zip(header, tokens):
                    if key in ('z', 'y', 'x', 'radius'):
                        val = float(val)
                    elif key in ('div_state',
                                 'cell_id', 'parent_id', 'track_id'):
                        val = int(val)
                    elif key == 't':
                        val = int(float(val))
                    if key == 'cell_id':
                        key = 'id'
                    elif key == 'radius':
                        key = 'r'
                    cell[key] = val
            else:
                cell['t'] = int(float(tokens[0]))
                cell['z'] = float(tokens[1])
                cell['y'] = float(tokens[2])
                cell['x'] = float(tokens[3])
                cell['id'] = int(tokens[4])

                if not points:
                    cell['parent_id'] = int(tokens[5])
                    cell['track_id'] = int(tokens[6])
                    if has_div_state and has_radius and has_name:
                        #   0  1  2  3  4        5          6         7  8      9
                        #   t, z, y, x, cell_id, parent_id, track_id, r, name, state = tokens
                        cell['r'] = float(tokens[7])
                        cell['name'] = tokens[8]
                        cell['state'] = int(tokens[9])
                    elif has_div_state and has_radius:
                        # t, z, y, x, cell_id, parent_id, track_id, r, state = tokens
                        cell['r'] = float(tokens[7])
                        cell['state'] = int(tokens[8])
                    elif has_div_state and has_name:
                        # t, z, y, x, cell_id, parent_id, track_id, name, state = tokens
                        cell['name'] = tokens[7]
                        cell['state'] = int(tokens[8])
                    elif has_radius and has_name:
                        # t, z, y, x, cell_id, parent_id, track_id, r, name = tokens
                        cell['r'] = float(tokens[7])
                        cell['name'] = tokens[8]
                    elif has_radius:
                        # t, z, y, x, cell_id, parent_id, track_id, r = tokens
                        cell['r'] = float(tokens[7])
                    elif has_div_state:
                        # t, z, y, x, cell_id, parent_id, track_id, state = tokens
                        cell['state'] = int(tokens[7])
                    elif has_name:
                        # t, z, y, x, cell_id, parent_id, track_id, name = tokens
                        cell['name'] = tokens[7]

            if frames and \
               (cell['t'] < frames[0] or cell['t'] >= frames[1]):
                continue
            position = [cell['t'], cell['z'], cell['y'], cell['x']]
            if min_dims is None:
                min_dims = position
            else:
                min_dims = [min(prev_min, curr)
                            for prev_min, curr in zip(min_dims, position)]

            if max_dims is None:
                max_dims = position
            else:
                max_dims = [max(prev_max, curr)
                            for prev_max, curr in zip(max_dims, position)]

            cell['score'] = 1
            nodes[cell['id']] = (cell['id'], cell)


    for cell_id, (_, cell) in nodes.items():
        parent_id = cell['parent_id']

        cell_loc = np.array([cell['z'],
                             cell['y'],
                             cell['x']])

        # fac = 3
        # dz = np.random.randint(-5*fac, 5*fac+1)
        # dy = np.random.randint(-5*fac, 5*fac+1)
        # dx = np.random.randint(-5*fac, 5*fac+1)
        # cell['z'] = cell_loc[0]-dz
        # cell['y'] = cell_loc[1]-dy
        # cell['x'] = cell_loc[2]-dx

        # cell_loc = np.array([cell['z'],
        #                      cell['y'],
        #                      cell['x']])

        if parent_id == -1:
            parent_vector = (0.0, 0.0, 0.0)
        else:
            parent_cell = nodes[parent_id][1]

            parent_cell_loc = np.array([parent_cell['z'],
                                        parent_cell['y'],
                                        parent_cell['x']])

            fac = 6
            dz = np.random.randint(-5*fac, 5*fac+1)
            dy = np.random.randint(-5*fac, 5*fac+1)
            dx = np.random.randint(-5*fac, 5*fac+1)
            parent_cell_loc = np.array([parent_cell['z']-dz,
                                        parent_cell['y']-dy,
                                        parent_cell['x']-dx])


            parent_vector = parent_cell_loc - cell_loc


        cell['parent_vector'] = tuple(parent_vector)
        # cell['parent_vector'] = (0, 0, 0)

        # rnd = np.random.rand()
        # if rnd < 0.9:
        #     if cell['div_state'] == 0:
        #         n = 0.9
        #         m = 0.05
        #         d = 0.05
        #     elif cell['div_state'] == 1:
        #         n = 0.05
        #         m = 0.9
        #         d = 0.05
        #     elif cell['div_state'] == 2:
        #         n = 0.05
        #         m = 0.05
        #         d = 0.9
        #     else:
        #         raise RuntimeError("")
        # elif rnd < 0.95:
        #     if cell['div_state'] == 0:
        #         n = 0.05
        #         m = 0.9
        #         d = 0.05
        #     elif cell['div_state'] == 1:
        #         n = 0.05
        #         m = 0.05
        #         d = 0.9
        #     elif cell['div_state'] == 2:
        #         n = 0.9
        #         m = 0.05
        #         d = 0.05
        #     else:
        #         raise RuntimeError("")
        # elif rnd <= 1.0:
        #     if cell['div_state'] == 0:
        #         n = 0.05
        #         m = 0.05
        #         d = 0.9
        #     elif cell['div_state'] == 1:
        #         n = 0.9
        #         m = 0.05
        #         d = 0.05
        #     elif cell['div_state'] == 2:
        #         n = 0.05
        #         m = 0.9
        #         d = 0.05
        #     else:
        #         raise RuntimeError("")
        # else:
        #     raise RuntimeError("")
        # cell['linajea_mouse_gt_as_rec_normal'] = n
        # cell['linajea_mouse_gt_as_rec_mother'] = m
        # cell['linajea_mouse_gt_as_rec_daughter'] = d
        # cell['linajea_140521_setup11_simple_early_middle_perfect_gt_as_rec_normal'] = float(cell['div_state'] == 0)
        # cell['linajea_140521_setup11_simple_early_middle_perfect_gt_as_rec_mother'] = float(cell['div_state'] == 1)
        # cell['linajea_140521_setup11_simple_early_middle_perfect_gt_as_rec_daughter'] = float(cell['div_state'] == 2)

        nodes[cell_id] = (cell_id, cell)

    nodes = nodes.values()

    if len(nodes) == 0:
        logger.error("Did not find any nodes in file %s" % tracks_file)
        return
    logger.info("Found %s nodes" % (len(nodes)))
    min_dims = Coordinate(min_dims)
    max_dims = Coordinate(max_dims)
    roi = Roi(min_dims, max_dims - min_dims + Coordinate((1, 1, 1, 1)))
    subgraph = db[roi]
    subgraph.add_nodes_from(nodes)
    logger.info("Added %s nodes"
                % (len(subgraph.nodes)))
    subgraph.write_nodes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, required=True,
                        help='input tracks file')
    parser.add_argument('--config', type=str,
                        help='config file with database host url')
    parser.add_argument('--db_name', type=str,
                        help="database name")
    parser.add_argument('--db_host', type=str, default=None,
                        help="database host")
    parser.add_argument('--has_div_state', action="store_true",
                        help='is division state in tracks file?')
    parser.add_argument("--has_radius", action="store_true",
                        help="is radius in tracks file?")
    parser.add_argument("--has_name", action="store_true",
                        help="is name/cell identity in tracks file?")
    parser.add_argument("--points", action="store_true",
                        help="only points in file, not tracks")
    parser.add_argument("-f", "--frames", type=int, nargs='*',
                        help="Frames to limit to")

    args = parser.parse_args()
    db_host = args.db_host if args.db_host is not None else load_config(args.config)['general']['db_host']
    if args.db_name:
        db_name = args.db_name
    else:
        raise RuntimeError("provide db name")
        # db_name = config['evaluate']['gt_db_name']
    print("creating database {}".format(db_name))
    add_tracks_to_mongo(args.tracks, db_host, db_name,
                        has_div_state=args.has_div_state,
                        has_radius=args.has_radius,
                        has_name=args.has_name,
                        frames=args.frames,
                        points=args.points)
