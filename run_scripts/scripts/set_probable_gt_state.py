import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pymongo
from scipy.optimize import linear_sum_assignment
import scipy.spatial
from sklearn import metrics
import toml

import gunpowder as gp

from linajea import (getNextInferenceData,
                     load_config,
                     parse_tracks_file)

logger = logging.getLogger(__name__)


def set_probable_gt_state(config, dry_run):
    sample = config.inference.data_source.datafile.filename
    if "polar" in sample:
        return
    logger.info("assigning probable gt state for %s, db: %s",
                sample, config.inference.data_source.db_name)

    if config.evaluate.parameters.roi:
        limit_to_roi = gp.Roi(offset=config.evaluate.parameters.roi.offset,
                          shape=config.evaluate.parameters.roi.shape)
    else:
        limit_to_roi = gp.Roi(offset=config.inference.data_source.roi.offset,
                              shape=config.inference.data_source.roi.shape)

    if limit_to_roi is not None:
        logger.info("limiting to roi {}".format(limit_to_roi))

    # read node candidates/pred_cells by frame
    pred_cells_by_frame, db_name, cells_db = get_pred_data(
        config, sample, limit_to_roi)

    # get gt cells
    location_by_frame, track_info_by_frame = get_gt_data(config, sample, limit_to_roi)

    matches_by_frame = compute_matching(config, pred_cells_by_frame, location_by_frame)

    compute_probable_gt_state(config, pred_cells_by_frame,
                              location_by_frame, track_info_by_frame,
                              matches_by_frame, dry_run, cells_db)
    if dry_run:
        logger.info("dry run, db %s not changed",
                    db_name)
    else:
        logger.info("inserted probable_gt_state into db %s",
                    db_name)

def get_pred_data(config, sample, limit_to_roi):

    client = pymongo.MongoClient(host=config.general.db_host)

    db_name = config.inference.data_source.db_name
    node_collection = 'nodes'

    db = client[db_name]
    cells_db = db[node_collection]
    cells = list(cells_db.find().sort('id', pymongo.ASCENDING))

    logger.info("host %s, db name %s, count %d",
                config.general.db_host,
                db_name, len(cells))

    pred_cells_by_frame = {}
    for cell in cells:
        if cell['score'] < config.inference.cell_score_threshold:
            continue
        cell_key = (cell['t'], cell['z'], cell['y'], cell['x'])
        if limit_to_roi is not None and \
           not limit_to_roi.contains(cell_key):
            # logger.info("cell %s outside of roi %s, skipping",
            #              cell_key, limit_to_roi)
            continue
        cell_value = None
        pred_cells_by_frame.setdefault(cell['t'], {})[cell_key] = cell_value
    logger.info("pred len %s",
                sum(map(len, pred_cells_by_frame.values())))

    return pred_cells_by_frame, db_name, cells_db


def get_gt_data(config, sample, limit_to_roi):
    if os.path.isdir(sample):
        data_config = load_config(
            os.path.join(sample, "data_config.toml"))
        filename_tracks = os.path.join(
            sample, data_config['general']['tracks_file'])
    else:
        data_config = load_config(
            os.path.join(os.path.dirname(sample), "data_config.toml"))
        if os.path.basename(sample) in data_config['general']['tracks_file']:
            filename_tracks = os.path.join(
                os.path.dirname(sample), data_config['general']['tracks_file'])
        else:
            filename_tracks = sample + "_" + data_config['general']['tracks_file']
    filename_tracks = os.path.splitext(filename_tracks)[0] + "_div_state.txt"

    voxel_size = np.array(config.inference.data_source.voxel_size)
    file_resolution = data_config['general']['resolution']
    if "Fluo" in filename_tracks:
        file_resolution = [1, 11, 1, 1]
    scale = np.array(voxel_size)/np.array(file_resolution)

    logger.info("scaling tracks by %s", scale)

    locations, track_info = parse_tracks_file(
        filename_tracks, limit_to_roi=limit_to_roi, scale=scale)
    logger.info("gt len %s", len(locations))

    location_by_frame = {}
    track_info_by_frame = {}
    for loc, tri in zip(locations, track_info):
        frame = int(loc[0])
        location_by_frame.setdefault(frame, []).append(loc)
        track_info_by_frame.setdefault(frame, []).append(tri)

    return location_by_frame, track_info_by_frame



def compute_matching(config, pred_cells_by_frame, location_by_frame):
    logger.info("#frames: %s", len(location_by_frame))
    logger.info("use hungarian matching")
    print(location_by_frame.keys(), pred_cells_by_frame.keys())
    matches_by_frame = {}
    for frame, gt_cells_fr in location_by_frame.items():
        if frame % 50 == 0:
            logger.info("frame %d", frame)
        if frame not in pred_cells_by_frame:
            logger.warning("frame %s missing in prediction!", frame)
            continue
        pred_cells_fr = list(pred_cells_by_frame[frame].keys())
        # scale_z = np.array((1, 0.5, 1, 1))
        scale_z = np.array((1.0, 1.0, 1.0, 1.0))
        gt_cells_fr_tmp = [p*scale_z for p in gt_cells_fr]
        pred_cells_fr_tmp = [p*scale_z for p in pred_cells_fr]

        if isinstance(config.evaluate.parameters.matching_threshold, dict):
            matching_threshold = -1
            for th, val in config.evaluate.parameters.matching_threshold.items():
                if th == -1 or frame < int(th):
                    matching_threshold = val
                    break
        else:
            matching_threshold = config.evaluate.parameters.matching_threshold
        costMat = np.zeros((len(gt_cells_fr), len(pred_cells_fr)),
                           dtype=np.float32)
        costMat[:,:] = matching_threshold

        gt_cells_tree = scipy.spatial.cKDTree(gt_cells_fr_tmp, leafsize=4)
        nn_distances, nn_locations = gt_cells_tree.query(
            pred_cells_fr_tmp, k=15,
            distance_upper_bound=matching_threshold)
        for dists, gIDs, pID in zip(nn_distances, nn_locations,
                                    range(len(pred_cells_fr))):
            for d, gID in zip(dists, gIDs):
                if d == np.inf:
                    continue
                costMat[gID, pID] = d

        pred_cells_tree = scipy.spatial.cKDTree(pred_cells_fr_tmp, leafsize=4)
        nn_distances, nn_locations = pred_cells_tree.query(
            gt_cells_fr_tmp, k=15,
            distance_upper_bound=matching_threshold)
        for dists, pIDs, gID in zip(nn_distances, nn_locations,
                                    range(len(gt_cells_fr))):
            for d, pID in zip(dists, pIDs):
                if d == np.inf:
                    continue
                if costMat[gID, pID] != matching_threshold:
                    assert abs(costMat[gID, pID] - d) <= 0.001, \
                        "non matching dist {} {}".format(
                            costMat[gID, pID], d)
                costMat[gID, pID] = d

        matches_by_frame[frame] = linear_sum_assignment(costMat)

    return matches_by_frame



def compute_probable_gt_state(config, pred_cells_by_frame,
                              location_by_frame, track_info_by_frame,
                              matches_by_frame, dry_run, cells_db):

    for frame in location_by_frame.keys():
        if frame % 50 == 0:
            logger.info("frame %d", frame)
        if frame not in pred_cells_by_frame:
            continue
        pred_cells_fr = list(pred_cells_by_frame[frame].keys())

        if isinstance(config.evaluate.parameters.matching_threshold, dict):
            matching_threshold = -1
            for th, val in config.evaluate.parameters.matching_threshold.items():
                if th == -1 or frame < int(th):
                    matching_threshold = val
                    break
        else:
            matching_threshold = config.evaluate.parameters.matching_threshold

        # scale_z = np.array((1, 0.5, 1, 1))
        scale_z = np.array((1.0, 1.0, 1.0, 1.0))
        for idx in range(len(matches_by_frame[frame][0])):
            gt_cell = matches_by_frame[frame][0][idx]
            if gt_cell == -1:
                continue
            gt_pos = location_by_frame[frame][gt_cell]
            gt_class = track_info_by_frame[frame][gt_cell][-1]
            gt_cell_id = track_info_by_frame[frame][gt_cell][0]

            pred_cell_fr_idx = matches_by_frame[frame][1][idx]
            pred_cell_pos = pred_cells_fr[pred_cell_fr_idx]

            gt_pos_tmp = gt_pos * scale_z
            pred_cell_pos_tmp = pred_cell_pos * scale_z
            dist = np.linalg.norm(
                np.array(gt_pos_tmp) -
                np.array(pred_cell_pos_tmp))

            if dist >= matching_threshold:
                continue

            query = {"t": pred_cell_pos[0],
                     "z": pred_cell_pos[1],
                     "y": pred_cell_pos[2],
                     "x": pred_cell_pos[3]}
            update = {"$set": {"probable_gt_state": int(gt_class),
                               "probable_gt_cell_id": gt_cell_id}}
            if not dry_run:
                logger.debug("updating %s with %s", query, update)
                cells_db.update_one(query, update)
            else:
                logger.debug("%s, %s", query, update)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--dry_run', action="store_true",
                        help='write to db?')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to process')
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    parser.add_argument('--val_param_id', type=int, default=None,
                        help='get test parameters from validation parameters_id')
    parser.add_argument('--param_id', type=int, default=None,
                        help='process parameters with parameters_id (e.g. resolve set of parameters)')
    parser.add_argument('--param_list_idx', type=str, default=None,
                        help='only solve idx parameter set in config')
    args = parser.parse_args()

    logging.basicConfig(level=20)

    for inf_config in getNextInferenceData(args, is_solve=True):
        set_probable_gt_state(inf_config, args.dry_run)
