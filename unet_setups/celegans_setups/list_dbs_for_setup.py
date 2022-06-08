from __future__ import absolute_import
import argparse
import logging
import os

import pymongo

from linajea.config import TrackingConfig

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config')
    parser.add_argument('--prefix', type=str, default="linajea_",
                        help='db name prefix')
    args = parser.parse_args()
    config = TrackingConfig.from_file(args.config)

    client = pymongo.MongoClient(host=config.general.db_host)
    for db_name in client.list_database_names():
        if not db_name.startswith(args.prefix):
            continue

        db = client[db_name]
        if "db_meta_info" not in db.list_collection_names():
            continue

        query_result = db["db_meta_info"].count_documents({})
        if query_result == 0:
            raise RuntimeError("invalid db_meta_info in db %s: no entry",
                               db_name)
        elif query_result > 1:
            raise RuntimeError(
                    "invalid db_meta_info in db %s: more than one entry (%d)",
                    db_name, query_result)
        else:
            assert query_result == 1
            query_result = db["db_meta_info"].find_one()
            if query_result["setup_dir"] == os.path.basename(config.general.setup_dir):
                print(db_name, query_result)
