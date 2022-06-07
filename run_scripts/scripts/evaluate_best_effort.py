import linajea
import linajea.tracking
import linajea.evaluation
import logging
import argparse
import time
from daisy import Roi

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def evaluate_best_effort(config, best_effort_key):
    db_name = config['general']['db_name']
    gt_db_name = config['evaluate']['gt_db_name']
    db_host = config['general']['db_host']
    data_dir = config['general']['data_dir']
    sample = config['general']['sample']
    matching_threshold = config['evaluate']['matching_threshold']
    sparse = config['evaluate']['sparse']
    frames = config['general'].get('frames', None)
    evaluate_config = config['general']
    evaluate_config.update(config['solve'])
    evaluate_config.update(config['evaluate'])
    evaluate_config.update({
        'model_type': 'nms',
        'max_cell_move': config['extract_edges']['edge_move_threshold']})

    start_time = time.time()
    edges_db = linajea.CandidateDatabase(
            db_name, db_host)
    edges_db.selected_key = best_effort_key

    logger.info("Reading cells and edges in db %s with parameter_id %s",
                db_name, best_effort_key)
    start_time = time.time()
    voxel_size, source_roi = linajea.get_source_roi(data_dir, sample)
    if frames:
        # limit source roi to frames
        limit_roi = Roi((frames[0], None, None, None),
                        (frames[1] - frames[0], None, None, None))
        source_roi = source_roi.intersect(limit_roi)
    subgraph = edges_db.get_selected_graph(source_roi)

    logger.info("Read %d cells and %d edges in %s seconds"
                % (subgraph.number_of_nodes(),
                   subgraph.number_of_edges(),
                   time.time() - start_time))

    if subgraph.number_of_edges() == 0:
        logger.warn("No selected edges for parameters_id %s. Skipping",
                    best_effort_key)
        return
    track_graph = linajea.tracking.TrackGraph(
        subgraph, frame_key='t', roi=subgraph.roi)

    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)

    logger.info("Reading ground truth cells and edges in db %s"
                % gt_db_name)

    gt_subgraph = gt_db[source_roi]
    logger.info("Read %d cells and %d edges in %s seconds"
                % (gt_subgraph.number_of_nodes(),
                   gt_subgraph.number_of_edges(),
                   time.time() - start_time))
    gt_track_graph = linajea.tracking.TrackGraph(
        gt_subgraph, frame_key='t', roi=gt_subgraph.roi)

    logger.info("Matching edges for parameters with id %s", best_effort_key)
    report = linajea.evaluation.evaluate(
            gt_track_graph,
            track_graph,
            matching_threshold=matching_threshold,
            sparse=sparse,
            validation_score=evaluate_config.get('validation_score', False),
            window_size=evaluate_config.get('window_size', 50))

    logger.info("Done evaluating results for %s. Saving results to mongo.",
                best_effort_key)
    edges_db.write_score(best_effort_key, report, frames=frames)
    logger.info("Done evaluating")
    linajea.print_time(time.time() - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="config file")
    parser.add_argument('best_effort_key', help="mongo key for best effort")
    args = parser.parse_args()
    config_file = args.config
    config = linajea.load_config(config_file)
    evaluate_best_effort(config, args.best_effort_key)
