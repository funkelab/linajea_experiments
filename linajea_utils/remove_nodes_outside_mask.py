import argparse
import pymongo
from tqdm import tqdm
from daisy import Roi, open_ds
import numpy as np


def remove_nodes_outside_mask(
        mongo_url, db_name, mask_file, time, mask_dataset="volumes/mask"):
    mask = open_ds(mask_file, mask_dataset, 'r')
    client = pymongo.MongoClient(mongo_url)
    db = client[db_name]
    collection = db['nodes']
    cursor = collection.find({'t': time})
    to_delete = []
    shape = (1, 3, 3, 3)
    for node in tqdm(cursor):
        offset = (node['t'], node['z'], node['y'], node['x'])
        roi = Roi(offset, shape)
        if not mask.roi.contains(roi):
            to_delete.append(node)
            continue
        mask_value = mask[roi].to_ndarray()
        if not np.any(mask_value):
            to_delete.append(node)
    print("Found %d nodes to delete in time %d" % (len(to_delete), time))
    result = collection.bulk_write(
            [pymongo.DeleteOne(n)
                for n in to_delete], ordered=False)
    print("Deleted %d nodes" % result.deleted_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo_url", type=str,
        default="localhost")
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--time", type=int)
    args = parser.parse_args()
    remove_nodes_outside_mask(args.mongo_url, args.db_name, args.data_file, args.time)
