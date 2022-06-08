import linajea
import linajea.tracking
import linajea.evaluation
from daisy import Roi, open_ds
import pymongo
import logging
import argparse
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fast_selection(mongo_url, db_name, selected_key):
    db = linajea.CandidateDatabase(db_name, mongo_url=mongo_url, mode='r+')
    selected_edges = \
        [[79, 81],
         [81, 82],
         [82, 83]]
    unselected_edges = \
        [[83, 84],
         [703, 704],
         [6419, 6420]]
    print("Storing selected edges {} and unselected edges {} in db {} with key {}"
          .format(selected_edges, unselected_edges, db_name, selected_key))
    db.store_edge_selections(selected_edges, unselected_edges, selected_key)
    print("Done storing selections")


def remove_nodes_below_threshold(mongo_url, db_name, threshold, dry_run):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    collection = db['nodes']
    if not dry_run:
        result = collection.bulk_write([pymongo.DeleteMany(
            {'score': {'$lt': threshold}})])
        print("Deleted %d nodes" % result.deleted_count)


def remove_nodes_outside_mask(
        mongo_url, db_name, mask_file, time, mask_dataset="volumes/mask",
        use_2d_mask=False, dry_run=False):
    mask = open_ds(mask_file, mask_dataset, 'r')
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    collection = db['nodes']
    cursor = collection.find({'t': time})
    to_delete = []
    shape = (1, 3, 3, 3)
    for node in tqdm(cursor):
        if use_2d_mask:
            offset = (node['y'], node['x'])
            roi = Roi(offset, shape[2:])
        else:
            offset = (node['t'], node['z'], node['y'], node['x'])
            roi = Roi(offset, shape)
        if not mask.roi.contains(roi):
            to_delete.append(node)
            continue
        mask_value = mask[roi].to_ndarray()
        if not np.any(mask_value):
            to_delete.append(node)
    print("Found %d nodes to delete in time %d" % (len(to_delete), time))
    if not dry_run:
        result = collection.bulk_write(
            [pymongo.DeleteOne(n)
             for n in to_delete], ordered=False)
        print("Deleted %d nodes" % result.deleted_count)


def add_ids_to_edges(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    collection = db['edges']
    collection.aggregate([
            {'$addFields': {'id': ['$source', '$target']}},
            {'$out': 'edges'}])


def remove_id_from_edges(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    collection = db['edges']
    collection.update_many({}, {'$unset': {'id': True}})


def remove_daisy_solve_colls(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    colls = db.list_collection_names()
    for coll in colls:
        if coll.startswith('solve'):
            print("dropping %s" % coll)
            print(db.drop_collection(coll))


def reset_solution(mongo_url, db_name, roi, selected_key):
    print("Resetting solution for database {}, key, {}, and roi {}"
          .format(db_name, selected_key, roi))
    db = linajea.CandidateDatabase(db_name, db_host=mongo_url, mode='r+')
    print("Reading edges")
    edges = db.read_edges(roi)
    num_removed = 0
    print("Starting loop over edges")
    for e in edges:
        if selected_key == 'all':
            for key in e.keys():
                if key.startswith('selected'):
                    del e[key]
                    # print("Dry run removing selected key {}".format(key))
            num_removed += 1
        else:
            if selected_key in e:
                del e[selected_key]
                num_removed += 1
    print("updating edges")
    db.update_edges(edges)
    print("Removed selection from {} edges".format(num_removed))


def drop_selected_colls(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    colls = db.list_collection_names()
    print(colls)
    for coll in colls:
        if coll.startswith('edges_'):
            print("dropping %s" % coll)
            db[coll].drop()


def remove_results(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    for collection in db.list_collection_names():
        if collection.startswith("results"):
            print("Dry run: Dropping %s" % collection)
            # db.drop_collection(collection)


def reset_all_solutions_except(mongo_url, db_name, selected_key):
    print("Resetting all solutions except {}".format(selected_key))
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    edges = db['edges']
    new_edges = []
    for edge in edges.find({}):
        copy = edge.copy()
        for key in edge.keys():
            if key.startswith('selected') and key != selected_key:
                del copy[key]
        new_edges.append(copy)
        logger.debug(copy)
    edges.bulk_write([
        pymongo.ReplaceOne(
            {
                'source': edge['source'],
                'target': edge['target']
            },
            edge
        )
        for edge in new_edges
    ])


def add_old_to_scores(mongo_url, db_name):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    scores = db['scores']
    for score in scores.find({}):
        _id = score['_id']
        if _id.startswith('selected'):
            score['_id'] = 'old_' + _id
            print(score)
            scores.delete_one({'_id': _id})
            scores.insert_one(score)


def print_dbs(host, username, password, port, replica_set):
    client = pymongo.MongoClient(
            host,
            username=username,
            password=password,
            replicaset=replica_set,
            port=port)
    dbs = client.list_databases()
    print("Data bases on host {}:".format(host))
    for db in dbs:
        print(db)


def create_mongo_url(host, username, password, port, replicaset):
    uri = "mongodb://{}:{}@{}:{}/admin?replicaSet={}".format(
            username, password, host, port, replicaset)
    print(uri)
    client = pymongo.MongoClient(uri)
    dbs = client.list_databases()
    print("Data bases on host {}:".format(host))
    for db in dbs:
        print(db)


def connect_via_url(mongo_url, db_name):
    db = linajea.CandidateDatabase(db_name, mongo_url)
    nodes = db.read_nodes(Roi((190, 0, 0, 0), (20, 4000, 2000, 2000)))
    print("Found {} nodes".format(len(nodes)))


def copy_db_to_new_host(old_host, new_url, db_name):
    new_client = pymongo.MongoClient(new_url)
    print("Starting to copy db {} from host {} to host {}"
          .format(db_name, old_host, new_url))
    new_client.admin.command(
            'copydb',
            fromdb=db_name,
            todb=db_name,
            fromhost=old_host)
    print("Done copying db")


def copy_roi_to_new_db(mongo_url, source_db, target_db, roi):
    source = linajea.CandidateDatabase(source_db, mongo_url)
    nodes = source.read_nodes(roi)
    edges = source.read_edges(roi, nodes=nodes)
    print("Found {} nodes and {} edges in roi {} and db {}"
          .format(len(nodes), len(edges), roi, source_db))
    target = linajea.CandidateDatabase(target_db, mongo_url, 'w')
    subgraph = target[roi]
    subgraph.write_nodes(nodes)
    print("Done writing nodes to {}".format(target_db))
    subgraph.write_edges(edges)
    print("Done writing edges to {}".format(target_db))


def filter_gt_and_copy_to_new_db(
        mongo_url, source_db, target_db, start_frame, end_frame):
    source = linajea.CandidateDatabase(source_db, mongo_url)
    roi = Roi((start_frame, 0, 0, 0),
              (end_frame - start_frame, 10000, 10000, 10000))
    subgraph = source[roi]
    remove_nodes = []
    remove_edges = []
    for edge in subgraph.edges:
        if 't' not in subgraph.nodes[edge[1]]:
            remove_edges.append(edge)
            remove_nodes.append(edge[1])
    subgraph.remove_edges_from(remove_edges)
    subgraph.remove_nodes_from(remove_nodes)

    track_graph = linajea.tracking.TrackGraph(subgraph, frame_key='t')
    tracks = track_graph.get_tracks()
    filtered_tracks = filter_ground_truth(
            tracks, start_frame, end_frame)
    target = linajea.CandidateDatabase(target_db, mongo_url, 'w')
    subgraph = target[roi]
    for track in filtered_tracks:
        subgraph.add_nodes_from(track.nodes(data=True))
        subgraph.add_edges_from(track.edges(data=True))
    subgraph.write_nodes()
    subgraph.write_edges()


def filter_ground_truth(gt_tracks, start_frame, end_frame):
    filtered_tracks = []
    for index, track in enumerate(gt_tracks):
        filter_flag = False
        for node, data in track.nodes(data=True):
            if start_frame is not None:
                if len(track.prev_edges(node)) == 0\
                        and data['t'] != start_frame:
                    print("Filtering track %d because node %d"
                          " in frame %d does not have a parent"
                          % (index, node, data['t']))
                    filter_flag = True
                    break
            if end_frame is not None:
                if len(track.next_edges(node)) == 0\
                        and data['t'] != end_frame - 1:
                    print("Filtering track %d because node %d "
                          "in frame %d does not have child"
                          % (index, node, data['t']))
                    filter_flag = True
                    break
        if not filter_flag:
            filtered_tracks.append(track)
    return filtered_tracks


def check_duplicate_node_ids(mongo_url, db):
    ids = set()
    client = pymongo.MongoClient(mongo_url)
    db = client[db]
    nodes = db['nodes']
    for node in nodes.find():
        _id = node['id']
        if _id in ids:
            print("Found duplicate node id %d" % _id)
        else:
            ids.add(_id)
    print("Done checking for duplicate node ids in database %s" % db)


def rename_db(mongo_url, old, new):
    client = pymongo.MongoClient(mongo_url)
    client.admin.command('copydb', fromdb=old, todb=new)
    client.drop_database(old)


def reset_exp(mongo_url, db_name, drop_steps, max_param_id, cls_key=None,
              dry_run=False):
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    colls = db.list_collection_names()
    if "predict" in drop_steps:
        coll = "nodes"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))
        coll = "predict_cells_daisy"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))

    if "edges" in drop_steps:
        coll = "edges"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))
        coll = "extract_edges_daisy"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))

    if "solve" in drop_steps:
        for coll in colls:
            if coll.startswith('solve'):
                print("dropping %s" % coll)
                if not dry_run:
                    print(db.drop_collection(coll))
        unset = {}
        for i in range(1, max_param_id+1):
            selected_key = 'selected_' + str(i)
            unset[selected_key] = ""

        print("unsetting selected_key")
        if not dry_run:
            db['edges'].update_many({}, {'$unset': unset})
        coll = "parameters"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))

    if "eval" in drop_steps:
        coll = "scores"
        print("dropping %s" % coll)
        if not dry_run:
            print(db.drop_collection(coll))

    if "class" in drop_steps:
        assert cls_key, "Please provide classifier key"
        unset = {}
        key = cls_key + "normal"
        unset[key] = ""
        key = cls_key + "mother"
        unset[key] = ""
        key = cls_key + "daughter"
        unset[key] = ""
        key = cls_key + "polar"
        unset[key] = ""
        unset['probable_gt_state'] = ""
        unset['probable_gt_cell_id'] = ""

        print("unsetting cls_key")
        if not dry_run:
            db['nodes'].update_many({}, {'$unset': unset})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function", type=str,
        help=("function to run. Options: \n"
              "rename_db\n"
              "remove_daisy_solve_colls\n"
              "remove_nodes_below_threshold\n"
              "remove_nodes_outside_mask\n"
              "reset_exp (in combination with drop)\n"))
    parser.add_argument(
        "--mongo_url", type=str,
        default="mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin")
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--dry_run", action="store_true",
                        help=("Don't execute db commands, not implemented for"
                              "all functions"))
    parser.add_argument("--new_db_name", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--cls_key", type=str)
    parser.add_argument("--time", type=int)
    parser.add_argument('-d', '--drop', dest='drop', default=[], nargs='+',
                        choices=['predict',
                                 'edges',
                                 'solve',
                                 'eval',
                                 'class',
                                 ],
                        help='Task to do for experiment.')
    parser.add_argument("--max_param_id", type=int, default=1000)
    parser.add_argument("--use_2d_mask", action="store_true",
                        help=("only consider xy coordinates when deciding to"
                              "remove nodes outside of mask"))
    args = parser.parse_args()
    if args.function == 'rename_db':
        old = args.db_name
        new = args.new_db_name
        print("Renaming %s to %s" % (old, new))
        rename_db(args.mongo_url, old, new)
    elif args.function == 'remove_daisy_solve_colls':
        assert args.db_name, "Please provide database name"
        remove_daisy_solve_colls(args.mongo_url, args.db_name)
    elif args.function == 'reset_exp':
        assert args.db_name, "Please provide database name"
        reset_exp(args.mongo_url, args.db_name, args.drop, args.max_param_id,
                  cls_key=args.cls_key, dry_run=args.dry_run)
    elif args.function == 'remove_nodes_below_threshold':
        assert args.db_name, "Please provide database name"
        assert args.threshold, "Please provide a threshold value"
        remove_nodes_below_threshold(args.mongo_url, args.db_name,
                                     args.threshold, dry_run=args.dry_run)
    elif args.function == 'remove_nodes_outside_mask':
        assert args.db_name, "Please provide database name"
        assert args.data_file, "Please provide a path to mask data file"
        remove_nodes_outside_mask(args.mongo_url, args.db_name, args.data_file,
                                  args.time, use_2d_mask=args.use_2d_mask,
                                  dry_run=args.dry_run)
