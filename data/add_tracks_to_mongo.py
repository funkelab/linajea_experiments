from linajea import CandidateDatabase, load_config
from daisy import Roi, Coordinate
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_tracks_to_mongo(tracks_file, db_host, db_name,
                        has_div_state=False, has_radius=False,
                        has_name=False, frames=None, points=False):

    db = CandidateDatabase(db_name, db_host, mode='w')
    nodes = []
    edges = []
    min_dims = None
    max_dims = None

    for line in open(tracks_file, 'r'):

        tokens = line.split()
        #   0  1  2  3  4        5          6         7  8      9
        if has_div_state and has_radius and has_name:
            t, z, y, x, cell_id, parent_id, track_id, r, name, state = tokens
        elif has_div_state and has_radius:
            t, z, y, x, cell_id, parent_id, track_id, r, state = tokens
        elif has_div_state and has_name:
            t, z, y, x, cell_id, parent_id, track_id, name, state = tokens
        elif has_radius and has_name:
            t, z, y, x, cell_id, parent_id, track_id, r, name = tokens
        elif has_radius:
            t, z, y, x, cell_id, parent_id, track_id, r = tokens
        elif has_div_state:
            t, z, y, x, cell_id, parent_id, track_id, state = tokens
        elif has_name:
            t, z, y, x, cell_id, parent_id, track_id, name = tokens
        elif points:
            t, x, y, z, cell_id = tokens
        else:
            t, z, y, x, cell_id, parent_id, track_id = tokens
        t = int(float(t))
        if frames and (t < frames[0] or t >= frames[1]):
            continue
        z = float(z)
        y = float(y)
        x = float(x)
        cell_id = int(cell_id)
        position = [t, z, y, x]
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

        nodes.append(
            (cell_id,
             {'t': t,
              'z': z,
              'y': y,
              'x': x,
              'id': cell_id,
              'score': 0}
             ))
        if not points:
            parent_id = int(parent_id)
            track_id = int(track_id)
            if has_div_state:
                nodes[-1][1]['state'] = int(state)
            if has_radius:
                nodes[-1][1]['r'] = float(r)
            if has_name:
                nodes[-1][1]['name'] = name

            if parent_id >= 0:
                if not frames or t > frames[0]:
                    edges.append(
                        (cell_id,
                         parent_id,
                         {'score': 0,
                          'distance': 0}
                         ))

    if len(nodes) == 0:
        logger.error("Did not find any nodes in file %s" % tracks_file)
        return
    logger.info("Found %s nodes and %s edges" % (len(nodes), len(edges)))
    min_dims = Coordinate(min_dims)
    max_dims = Coordinate(max_dims)
    roi = Roi(min_dims, max_dims - min_dims + Coordinate((1, 1, 1, 1)))
    subgraph = db[roi]
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)
    logger.info("Added %s nodes and %s edges"
                % (len(subgraph.nodes), len(subgraph.edges)))
    subgraph.write_nodes()
    subgraph.write_edges()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, required=True,
                        help='input tracks file')
    parser.add_argument('--config', type=str, required=True,
                        help='config file with database host url')
    parser.add_argument('--db_name', type=str,
                        help="database name")
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
    config = load_config(args.config)
    db_host = config['general']['db_host']
    if args.db_name:
        db_name = args.db_name
    else:
        db_name = config['evaluate']['gt_db_name']
    print("creating database {}".format(db_name))
    add_tracks_to_mongo(args.tracks, db_host, db_name,
                        has_div_state=args.has_div_state,
                        has_radius=args.has_radius,
                        has_name=args.has_name,
                        frames=args.frames,
                        points=args.points)
