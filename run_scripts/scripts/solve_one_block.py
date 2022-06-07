from linajea import (
        CandidateDatabase,
        load_config,
        tracking_params_from_config)
from linajea.tracking import track
import time
import logging
import daisy
import sys
import pylp

pylp.verbose = True

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def solve_in_block(
        db_host,
        db_name,
        parameters,
        block,
        parameters_id,
        solution_roi=None,
        cell_cycle_key=None):
    # Solution_roi is the total roi that you want a solution in
    # Limiting the block to the solution_roi allows you to solve
    # all the way to the edge, without worrying about reading
    # data from outside the solution roi
    # or paying the appear or disappear costs unnecessarily

    logger.debug("Solving in block %s", block)

    if solution_roi:
        # Limit block to source_roi
        logger.debug("Block write roi: %s", block.write_roi)
        logger.debug("Solution roi: %s", solution_roi)
        read_roi = block.read_roi.intersect(solution_roi)
        write_roi = block.write_roi.intersect(solution_roi)
    else:
        read_roi = block.read_roi
        write_roi = block.write_roi

    logger.debug("Write roi: %s", str(write_roi))

    graph_provider = CandidateDatabase(
        db_name,
        db_host,
        mode='r+')
    start_time = time.time()
    selected_keys = ['selected_' + str(pid) for pid in parameters_id]
    edge_attrs = selected_keys.copy()
    edge_attrs.extend(["prediction_distance", "distance"])
    graph = graph_provider.get_graph(
            read_roi,
            edge_attrs=edge_attrs
            )

    # remove dangling nodes and edges
    dangling_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if 't' not in data
    ]
    graph.remove_nodes_from(dangling_nodes)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    logger.info("Reading graph with %d nodes and %d edges took %s seconds"
                % (num_nodes, num_edges, time.time() - start_time))

    if num_edges == 0:
        logger.info("No edges in roi %s. Skipping"
                    % read_roi)
        # write_done(block, step_name, db_name, db_host)
        return 0

    frames = [read_roi.get_offset()[0],
              read_roi.get_offset()[0] + read_roi.get_shape()[0]]
    track(graph, parameters, selected_keys, frames=frames,
          cell_cycle_key=cell_cycle_key)
    #  start_time = time.time()
    #  graph.update_edge_attrs(
    #          write_roi,
    #          attributes=selected_keys)
    #  logger.info("Updating %d keys for %d edges took %s seconds"
    #              % (len(selected_keys),
    #                 num_edges,
    #                 time.time() - start_time))
    # write_done(block, step_name, db_name, db_host)
    return 0


if __name__ == '__main__':
    block = daisy.Block(
            block_id=338350,
            read_roi=daisy.Roi((245, -100, -100, 900),
                               (9, 700, 700, 700)),
            write_roi=daisy.Roi((245, 0, 0, 1000),
                                (5, 500, 500, 500)),
            total_roi=None)
    config = load_config(sys.argv[1])
    db_host = config['general']['db_host']
    db_name = config['general']['db_name']
    parameters = [tracking_params_from_config(config)]
    parameters_id = [246]
    cell_cycle_key = config['solve']['cell_cycle_key']
    solve_in_block(
        db_host,
        db_name,
        parameters,
        block,
        parameters_id,
        cell_cycle_key=cell_cycle_key)
