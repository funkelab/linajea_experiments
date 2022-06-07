import linajea
import linajea.tracking
import linajea.evaluation
import time
import daisy
import logging
import pandas
import argparse
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_tracks_to_csv(
        tracks,
        out_file,
        header=True):
    for track in tracks:
        remove_nodes = []
        for node, data in track.nodes(data=True):
            if 't' not in data:
                remove_nodes.append(node)
        track.remove_nodes_from(remove_nodes)

    fields = ['t', 'z', 'y', 'x',
              'cell_id', 'parent_id', 'track_id',
              'node_score', 'edge_score']
    delim = ', '
    # (world coordiantes)
    with open(out_file, 'w') as f:
        if header:
            # write header
            f.write(delim.join(fields))
            f.write('\n')

        for track_id, track in enumerate(tracks):
            for node_id, data in track.nodes(data=True):
                t = data['t']
                z = data['z']
                y = data['y']
                x = data['x']
                parents = list(track.prev_edges(node_id))
                assert len(parents) < 2, "nodes has %d parents" % len(parents)
                if len(parents) == 0:
                    parent_id = -1
                    edge_score = 0.0
                else:
                    parent_id = parents[0][1]
                    edge = track.edges[(node_id, parent_id)]
                    edge_score = edge['prediction_distance'] if\
                        'prediction_distance' in edge else 0
                node_score = data['score']

                f.write(delim.join(list(map(str, [
                    t, z, y, x, node_id,
                    parent_id, track_id, node_score, edge_score]))))
                f.write('\n')


def write_points_with_parent_vectors(tracks, out_file):
    for track in tracks:
        remove_nodes = []
        for node, data in track.nodes(data=True):
            if 't' not in data:
                remove_nodes.append(node)
        track.remove_nodes_from(remove_nodes)

    # node_id, t, z, y, x, pv_z, pv_y, pv_x (world coordiantes)
    with open(out_file, 'w') as f:
        for track in tracks:
            for node_id, data in track.nodes(data=True):
                t = data['t']
                z = data['z']
                y = data['y']
                x = data['x']
                pv_z, pv_y, pv_x = data['parent_vector']
                f.write("%d, %d, %d, %d, %d, %f, %f, %f\n"
                        % (node_id, t, z, y, x, pv_z, pv_y, pv_x))


def get_tracks(
        db_name,
        frames,
        parameters_id):
    if frames is not None:
        start_frame, end_frame = frames
        assert(start_frame < end_frame)
        num_frames = end_frame - start_frame
    else:
        start_frame = None
        num_frames = None

    # smaller debug roi:
    roi = daisy.Roi((start_frame, 0, 0, 0), (num_frames, 1e10, 1e10, 1e10))

    mongo_url = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    db = linajea.CandidateDatabase(
            db_name, mongo_url, parameters_id=parameters_id)
    logger.info("Reading cells and edges in %s" % roi)
    if parameters_id is None:
        subgraph = db[roi]
    else:
        subgraph = db.get_selected_graph(
                roi, edge_attrs=['prediction_distance'])
    track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
    logger.info("Found %d nodes and %d edges db %s"
                % (track_graph.number_of_nodes(),
                   track_graph.number_of_edges(),
                   db_name))
    tracks = track_graph.get_tracks()
    return tracks


def get_best_tracks(
        db_name,
        frames,
        vald_db_name,
        vald_frames):
    if frames is not None:
        start_frame, end_frame = frames
        assert(start_frame < end_frame)
        num_frames = end_frame - start_frame
    else:
        start_frame = None
        num_frames = None

    roi = daisy.Roi((start_frame, 0, 0, 0), (num_frames, 1e10, 1e10, 1e10))
    mongo_url = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    vald_db = linajea.CandidateDatabase(vald_db_name, mongo_url, 'r')
    scores = vald_db.get_scores(frames=vald_frames,
                                filters={'version': 'v1.3-dev'})
    if len(scores) == 0:
        logger.error("NO VALD SCORES FOUND!")
        return None
    dataframe = pandas.DataFrame(scores)
    logger.info("dataframe:")
    logger.info(dataframe)
    logger.debug("data types of dataframe columns: %s"
                 % str(dataframe.dtypes))
    score_columns = ['fn_edges', 'identity_switches',
                     'fp_divisions', 'fn_divisions']
    dataframe['sum_errors'] = sum([dataframe[col] for col in score_columns])
    dataframe.sort_values('sum_errors', inplace=True)
    best_result = dataframe.iloc[0].to_dict()
    for key, value in best_result.items():
        try:
            best_result[key] = value.item()
        except AttributeError:
            pass
    best_params = linajea.tracking.TrackingParameters(**best_result)

    db = linajea.CandidateDatabase(
            db_name, mongo_url)
    best_id = db.get_parameters_id(best_params, fail_if_not_exists=True)
    logger.info("using id %d in db %s", best_id, db_name)
    db.set_parameters_id(best_id)
    logger.info("Reading cells and edges in %s" % roi)
    subgraph = db.get_selected_graph(
            roi, edge_attrs=['prediction_distance'])
    track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
    logger.info("Found %d nodes and %d edges db %s"
                % (track_graph.number_of_nodes(),
                   track_graph.number_of_edges(),
                   db_name))
    tracks = track_graph.get_tracks()
    return tracks


def get_matched_tracks(
        db_name,
        frames,
        parameters_id,
        gt_db_name):
    if frames is not None:
        start_frame, end_frame = frames
        assert(start_frame < end_frame)
        num_frames = end_frame - start_frame
    else:
        start_frame = None
        num_frames = None

    # smaller debug roi:
    roi = daisy.Roi((start_frame, 0, 0, 0), (num_frames, 1e10, 1e10, 1e10))

    mongo_url = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    db = linajea.CandidateDatabase(
            db_name, mongo_url, parameters_id=parameters_id)
    logger.info("Reading cells and edges in %s" % roi)
    if parameters_id is None:
        subgraph = db[roi]
    else:
        subgraph = db.get_selected_graph(roi)
    track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
    logger.info("Found %d nodes and %d edges from db %s"
                % (track_graph.number_of_nodes(),
                   track_graph.number_of_edges(),
                   db_name))

    gt_db = linajea.CandidateDatabase(gt_db_name, mongo_url)

    logger.info("Reading ground truth cells and edges in db %s"
                % gt_db_name)
    start_time = time.time()
    gt_subgraph = gt_db[roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))
    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    gt_edges, rec_edges, edge_matches, edge_fps =\
        linajea.evaluation.match_edges(
            gt_track_graph,
            track_graph,
            matching_threshold=25)

    tracks = track_graph.get_tracks()
    logger.info("Found %d tracks. Filtering" % len(tracks))
    filtered_tracks = []
    matched_edges = set(rec_edges[rec_edge_index]
                        for _, rec_edge_index in edge_matches)
    for track in tracks:
        for edge in track.edges():
            if edge in matched_edges:
                filtered_tracks.append(track)
                break
    logger.info("Done filtering. Returning %d tracks" % len(filtered_tracks))
    return filtered_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('-f', '--frames', type=int, nargs=2, default=None)
    parser.add_argument('out_file')
    parser.add_argument('-id', '--parameters_id', type=int)
    parser.add_argument('-gt', '--gt-db-name', default=None)
    parser.add_argument('-v', '--vald-db-name', default=None)
    parser.add_argument('-vf', '--vald-frames', type=int, nargs=2)
    parser.add_argument('-p', '--points', action='store_true')
    parser.add_argument('-nh', '--no-header', action='store_true')
    parser.add_argument('-ml', '--min-length', type=int, default=None)
    args = parser.parse_args()

    db_name = args.db_name  # the database to get the results from
    out_file = args.out_file

    points = args.points
    parameters_id = args.parameters_id
    gt_db_name = args.gt_db_name

    start_time = time.time()
    if args.vald_db_name is not None:
        tracks = get_best_tracks(
                db_name,
                args.frames,
                args.vald_db_name,
                args.vald_frames)

    elif gt_db_name is not None:
        tracks = get_matched_tracks(
                db_name,
                args.frames,
                parameters_id,
                gt_db_name)
    else:
        tracks = get_tracks(
                db_name,
                args.frames,
                parameters_id)
    if tracks is None:
        sys.exit()

    if args.min_length:
        tracks = [t for t in tracks if len(t) >= args.min_length]

    if points:
        logger.info("Writing points from %d tracks to %s"
                    % (len(tracks), out_file))
        write_points_with_parent_vectors(tracks, out_file)
    else:
        logger.info("Writing %d tracks to %s" % (len(tracks), out_file))
        write_tracks_to_csv(tracks, out_file, header=not args.no_header)
    end_time = time.time()
    linajea.print_time(end_time - start_time)
