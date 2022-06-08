import argparse
import logging
import time
import torch
from vgg_model import get_cell_cycle_model
from torch.nn.functional import softmax
import toml
import os
import numpy as np
from mongo_data import MongoCandidates
from data_loader import get_data_loader
import pymongo

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def predict(
        model,
        data_loader,
        normalize=False):
    # expects model already loaded onto device with state set
    model.eval()
    # store predictions
    start = time.time()
    ids = []
    logits = []
    probabilities = []
    t_predict = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sample in data_loader:
        batch_ids = sample['id']
        ids.extend(batch_ids)
        data = sample['data']
        data = data.to(device)
        t0 = time.time()
        prediction = model(data)
        logger.debug("logits: %s", str(prediction))
        normalized = softmax(prediction)
        logger.debug("probabilities: %s", str(normalized))
        t_predict += time.time() - t0
        # Iterate over batch and grab predictions
        for k in range(np.shape(prediction)[0]):
            probs = normalized[k, :].tolist()
            ls = prediction[k, :].tolist()
            probabilities.append(probs)
            logits.append(ls)

    total_time = time.time() - start
    print("total time: %d" % total_time)
    if normalize:
        return ids, probabilities
    else:
        return ids, logits


def predict_from_db(config, iteration, frames, overwrite=False):
    db_name = config['validate']['candidate_db']
    db_host = config['db_host']
    data_file = config['data_file']
    roi_size = config['input_shape']
    label_name = 'vgg_' + str(iteration)

    model = get_cell_cycle_model(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkname = os.path.join(config['setup_dir'],
                             'model_checkpoint_%d' % iteration)
    checkpoint = torch.load(checkname, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    mongo_candidates = MongoCandidates(
            db_host,
            db_name,
            data_file,
            roi_size,
            label_name,
            1,   # n_gpus
            0,   # gpu_id
            frames=frames,
            overwrite=overwrite)
    data_loader = get_data_loader(
            mongo_candidates,
            config['batch_size'],
            roi_size)
    ids, prediction = predict(model, data_loader)
    write_predictions_to_db(db_name, db_host, ids, prediction, label_name)


def write_predictions_to_db(
        db_name, db_host,
        ids, predictions,
        label_name):
    if len(predictions) == 0:
        logger.info("No predictions to write: returning")
        return
    start_time = time.time()
    logger.info("Writing %d predictions to db %s", len(ids), db_name)
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    node_collection = db['nodes']
    updates = []
    for node_id, prediction in zip(ids, predictions):
        if node_id is None:
            continue
        _filter = {'id': int(np.int64(node_id))}
        update = {'$set': {label_name: list(prediction)}}
        updates.append(pymongo.UpdateOne(_filter, update))
    node_collection.bulk_write(updates, ordered=False)
    end_time = time.time() - start_time
    logger.info("Writing predictions took %d seconds", end_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file', required=True)
    parser.add_argument('-i', '--iteration', type=int, required=True)
    parser.add_argument('-f', '--frames', type=int, nargs=2, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    with open(args.config) as f:
        config = toml.load(f)

    logger.info("Config: %s", config)
    predict_from_db(config, args.iteration, args.frames,
                    overwrite=args.overwrite)
