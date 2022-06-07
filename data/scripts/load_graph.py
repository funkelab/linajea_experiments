import linajea
import networkx as nx
import daisy
import logging

logger = logging.getLogger(__name__)


def load_graph_from_db(
        db_name,
        db_host,
        start_frame,
        end_frame,
        key=None):

    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 1000000, 1000000, 1000000))
    db = linajea.CandidateDatabase(db_name, db_host)
    if key is None:
        graph = db[roi]
    else:
        graph = db.get_graph(
                roi,
                edges_filter={key: True},
                edge_attrs=[key])
    return graph


def read_nodes_and_edges(filename):
    logger.info("Reading nodes and edges from %s" % filename)
    nodes = {}
    edges = []
    max_id = -1
    for line in open(filename):
        # 0  1  2  3  4        5          6
        # t, z, y, x, cell_id, parent_id, track_id
        t, z, y, x, cell_id, parent_id, track_id = line.split()
        if t == 't':
            print("File has header - skipping")
            continue
        t = int(float(t))
        z = float(z)
        y = float(y)
        x = float(x)
        cell_id = int(cell_id)
        parent_id = int(parent_id)
        track_id = int(track_id)
        if parent_id > 0:
            edges.append(
                {
                    'source': cell_id,
                    'target': parent_id,
                })

        nodes[cell_id] = {
                'position': (t, z, y, x),
                't': t,
                'z': z,
                'y': y,
                'x': x,
                'id': cell_id,
            }
        max_id = max(max_id, cell_id)
    logger.info("%d nodes read" % len(nodes))
    logger.info("%d edges read" % len(edges))
    logger.info("Max cell id is %d" % max_id)

    return nodes, edges


def load_graph_from_file(filename):
    nodes, edges = read_nodes_and_edges(filename)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.items())
    graph.add_edges_from([(edge['source'], edge['target'])
                          for edge in edges])
    return graph
