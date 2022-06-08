import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import numpy as np
import pymongo
from scipy.optimize import linear_sum_assignment
import scipy.spatial
from sklearn import metrics

import gunpowder as gp
from linajea import (checkOrCreateDB,
                     load_config,
                     parse_tracks_file)

logger = logging.getLogger(__name__)


def eval_node_candidates(config, set_gt_db):
    sample = config.inference.data_source.datafile.filename
    logger.info("evaluating %s with threshold %s", sample,
                config.inference.prob_threshold)

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
        config, sample, limit_to_roi, set_gt_db)

    # get gt cells
    location_by_frame, track_info_by_frame, gt_parent_map, gt_children_map, \
        gt_idx_map = get_gt_data(config, sample, limit_to_roi)

    if config.model.with_polar:
        pred_cells_by_frame_p, _, _ = get_pred_data(
            config, sample + "/polar", limit_to_roi, set_gt_db)
        for fr, cells in pred_cells_by_frame_p.items():
            pred_cells_by_frame[fr].update(cells)

        id_offset = max(gt_idx_map.keys()) + 1
        logger.info("id offset for polar cells %s", id_offset)
        location_by_frame_p, track_info_by_frame_p, gt_parent_map_p, \
            gt_children_map_p, gt_idx_map_p = get_gt_data(
                config, sample + "/polar", limit_to_roi, id_offset=id_offset)
        for fr in location_by_frame_p.keys():
            location_by_frame[fr] += location_by_frame_p[fr]
            track_info_by_frame[fr] += track_info_by_frame_p[fr]
        for idx in gt_idx_map:
            if idx in gt_idx_map_p:
                raise RuntimeError("double entry in gt_idx_map %s %s %s", idx,
                                   gt_idx_map[idx], gt_idx_map_p[idx])
        for idx in gt_parent_map:
            if idx in gt_parent_map_p:
                raise RuntimeError(
                    "double entry in gt_parent_map %s %s %s",
                    idx, gt_parent_map[idx], gt_parent_map_p[idx])
        for idx in gt_children_map:
            if idx in gt_children_map_p:
                raise RuntimeError(
                    "double entry in gt_children_map %s %s %s", idx,
                    gt_children_map[idx], gt_children_map_p[idx])
        gt_idx_map.update(gt_idx_map_p)
        gt_parent_map.update(gt_parent_map_p)
        gt_children_map.update(gt_children_map_p)

    matches_by_frame = compute_matching(config, pred_cells_by_frame,
                                        location_by_frame)

    if config.evaluate.find_fn:
        return None

    results = compute_metrics(config, pred_cells_by_frame,
                              location_by_frame, track_info_by_frame,
                              gt_parent_map, gt_children_map, gt_idx_map,
                              matches_by_frame, set_gt_db, cells_db)
    if set_gt_db:
        if config.evaluate.dry_run:
            logger.info("dry run, db %s not changed",
                        db_name)
        else:
            logger.info("inserted probable_gt_state into db %s",
                        db_name)

    results_out = compress_and_print_results(config, results)
    return results_out


def get_pred_data(config, sample, limit_to_roi, set_gt_db):
    classes = list(config.model.classes)
    class_ids = list(config.model.class_ids)

    client = pymongo.MongoClient(host=config.general.db_host)

    if config.inference.use_database:
        db_name = config.inference.data_source.db_name
        node_collection = 'nodes'
        if config.predict.prefix:
            prefix = config.predict.prefix
        else:
            prefix = os.path.join(
                os.path.basename(config.general.setup_dir).split(
                    "classifier_")[-1],
                "test", str(config.inference.checkpoint) + "_")
    else:
        config.evaluate.parameters.matching_threshold[-1] = 1
        db_name = \
            "linajea_celegans_" + os.path.basename(config.general.setup_dir)
        if os.path.basename(sample) == "polar":
            name = os.path.basename(os.path.dirname(sample)) + "_" + \
                os.path.basename(sample)
        else:
            name = os.path.basename(sample)
        node_collection = "nodes_" + name +  \
            ("_w_swa" if config.predict.use_swa else "_wo_swa") + \
            ("_ttr{}".format(config.predict.test_time_reps)
             if config.predict.test_time_reps > 1 else "") + \
            ("_roi{}".format(config.inference.data_source.roi.shape[0])
             if config.inference.data_source.roi.shape[0] != 200 else "") + \
            "_" + str(config.inference.checkpoint)
        prefix = config.predict.prefix

    db = client[db_name]
    cells_db = db[node_collection]
    cells = list(cells_db.find().sort('id', pymongo.ASCENDING))

    logger.info("host %s, db name %s, count %d",
                config.general.db_host,
                db_name, len(cells))

    pred_cells_by_frame = {}
    for cell in cells:
        if config.inference.use_database and \
           cell['score'] < config.inference.db_meta_info.cell_score_threshold:
            continue
        cell_key = (cell['t'], cell['z'], cell['y'], cell['x'])
        if limit_to_roi is not None and \
           not limit_to_roi.contains(cell_key):
            continue
        try:
            cell_value = (cell[prefix+classes[class_ids.index(0)]],
                          cell[prefix+classes[class_ids.index(1)]],
                          cell[prefix+classes[class_ids.index(2)]])
            if config.model.with_polar:
                cell_value = cell_value + (cell[prefix+"polar"],)
        except:
            if config.inference.use_database and \
               set_gt_db:
                cell_value = None
            else:
                logger.info("%s", cell)
                raise RuntimeError("")
        pred_cells_by_frame.setdefault(cell['t'], {})[cell_key] = cell_value
    logger.info("pred len %s",
                sum(map(len, pred_cells_by_frame.values())))

    return pred_cells_by_frame, db_name, cells_db


def get_gt_data(config, sample, limit_to_roi, id_offset=0):
    is_polar = "polar" in sample
    if is_polar:
        sample = sample.replace("/polar", "")

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
    if is_polar:
        filename_tracks = os.path.splitext(filename_tracks)[0] + "_polar.txt"
    else:
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

    gt_idx_map = {}
    gt_parent_map = {}
    gt_children_map = {}
    if is_polar:
        for loc, meta in zip(locations, track_info):
            gt_idx_map[meta[0]+id_offset] = (loc, max(config.model.class_ids) + 1)
    else:
        for loc, meta in zip(locations, track_info):
            div_state = meta[-1]
            gt_idx_map[meta[0]+id_offset] = (loc, div_state)
    for loc, meta in zip(locations, track_info):
        parent_id = meta[1]
        if parent_id == -1:
            continue
        gt_parent_map[tuple([l for l in loc.ravel()])] = parent_id+id_offset \
            if parent_id > 0 else None
        cur_idx = meta[0]+id_offset
        try:
            parent_loc = gt_idx_map[parent_id+id_offset][0]
        except KeyError:
            logger.warning("parent removed by limit roi? %s, %s",
                           parent_id, loc)
            continue
        gt_children_map.setdefault(
            tuple([l for l in parent_loc.ravel()]), []).append(cur_idx)

    location_by_frame = {}
    track_info_by_frame = {}
    for loc, tri in zip(locations, track_info):
        frame = int(loc[0])
        location_by_frame.setdefault(frame, []).append(loc)
        if is_polar:
            tri = list(tri)
            tri.append(max(config.model.class_ids) + 1)
            tri = np.array(tri, dtype=object)
        track_info_by_frame.setdefault(frame, []).append(tri)

    return location_by_frame, track_info_by_frame, \
        gt_parent_map, gt_children_map, gt_idx_map


def compute_matching(config, pred_cells_by_frame, location_by_frame):
    logger.info("#frames: %s", len(location_by_frame))
    logger.info("use hungarian matching")
    matches_by_frame = {}
    for frame, gt_cells_fr in location_by_frame.items():
        if frame % 50 == 0:
            logger.info("frame %d", frame)
        if frame not in pred_cells_by_frame:
            logger.warning("frame %s missing in prediction!", frame)
            continue
        pred_cells_fr = list(pred_cells_by_frame[frame].keys())

        for th, val in config.evaluate.parameters.matching_threshold.items():
            if th == -1 or frame < int(th):
                matching_threshold = val
                break
        costMat = np.zeros((len(gt_cells_fr), len(pred_cells_fr)),
                           dtype=np.float32)
        costMat[:, :] = matching_threshold

        gt_cells_tree = scipy.spatial.cKDTree(gt_cells_fr, leafsize=4)
        nn_distances, nn_locations = gt_cells_tree.query(
            pred_cells_fr, k=15,
            distance_upper_bound=matching_threshold)
        for dists, gIDs, pID in zip(nn_distances, nn_locations,
                                    range(len(pred_cells_fr))):
            for d, gID in zip(dists, gIDs):
                if d == np.inf:
                    continue
                costMat[gID, pID] = d

        pred_cells_tree = scipy.spatial.cKDTree(pred_cells_fr, leafsize=4)
        nn_distances, nn_locations = pred_cells_tree.query(
            gt_cells_fr, k=15,
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
        if config.evaluate.find_fn:
            fns = np.setdiff1d(list(range(len(gt_cells_fr))),
                               matches_by_frame[frame][0])
            logger.info(("frame: {}, count gt: {}, count pred: {}, "
                         "count matches: {}, false negatives ({}): {}").format(
                frame,
                len(gt_cells_fr), len(pred_cells_fr),
                len(matches_by_frame[frame][0]), len(fns), fns))

            # for debugging
            out_fr = -1
            if frame == out_fr:
                write_matches_frame(frame, gt_cells_fr, pred_cells_fr,
                                    matches_by_frame, matching_threshold)
    return matches_by_frame


def write_matches_frame(frame, gt_cells_fr, pred_cells_fr,
                        matches_by_frame, matching_threshold):
    raise RuntimeError("check scaling vs matching threshold")
    it = 400000
    gts = list(range(len(gt_cells_fr)))
    ps = list(range(len(pred_cells_fr)))
    mts = open("matches_{}_fr{}.txt".format(it, frame), 'w')
    mts_ms_p = open("matches_{}_fr{}_ms_p.txt".format(it, frame), 'w')
    mts_ms_gt = open("matches_{}_fr{}_ms_gt.txt".format(it, frame), 'w')
    gts_m = []
    p_m = []

    for ma_gt, ma_p in zip(matches_by_frame[frame][0],
                           matches_by_frame[frame][1]):
        gt_pos = gt_cells_fr[ma_gt]
        p_pos = pred_cells_fr[ma_p]
        if np.linalg.norm(gt_pos-p_pos) >= matching_threshold:
            logger.info("%s", gt_pos)
            logger.info("%s", p_pos)
            logger.info("%s", np.linalg.norm(gt_pos-p_pos))
            continue
        gts_m.append(ma_gt)
        p_m.append(ma_p)

        mts.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(
            gt_pos[0], gt_pos[1], gt_pos[2], gt_pos[3],
            p_pos[0], p_pos[1], p_pos[2], p_pos[3]))
    mts.close()
    gts = np.setdiff1d(gts, gts_m)
    logger.info("%s", ps)
    ps = np.setdiff1d(ps, p_m)
    logger.info("%s", p_m)
    for g in gts:
        gt_pos = gt_cells_fr[g]
        mts_ms_gt.write("{}, {}, {}, {}\n".format(
            gt_pos[0], gt_pos[1], gt_pos[2], gt_pos[3]))
    mts_ms_gt.close()
    for p in ps:
        p_pos = pred_cells_fr[p]
        mts_ms_p.write("{}, {}, {}, {}\n".format(
            p_pos[0], p_pos[1], p_pos[2], p_pos[3]))
    mts_ms_p.close()


def compute_metrics(config, pred_cells_by_frame,
                    location_by_frame, track_info_by_frame,
                    gt_parent_map, gt_children_map, gt_idx_map,
                    matches_by_frame, set_gt_db, cells_db):

    results = {}
    class_ids = list(config.model.class_ids)
    if config.model.with_polar:
        class_ids.append(3)
    for idx in class_ids:
        results[idx] = {
            'gt': 0,
            'pred': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    normal_class_id = 0
    mother_class_id = 1
    daughter_class_id = 2
    list_gt_class = []
    list_pred_class = []
    for frame in location_by_frame.keys():
        if frame % 50 == 0:
            logger.info("frame %d", frame)
        if frame not in pred_cells_by_frame:
            for gt_cell in track_info_by_frame[frame]:
                gt_class = gt_cell[-1]
                results[gt_class]['fn'] += 1
                results[gt_class]['gt'] += 1
            logger.info("frame {} not in pred".format(frame))
            continue
        pred_cells_fr = list(pred_cells_by_frame[frame].keys())

        for th, val in config.evaluate.parameters.matching_threshold.items():
            if th == -1 or frame < int(th):
                matching_threshold = val
                break
        num_matches = 0
        invalid_dist_too_high_match_gt = []
        for idx in range(len(matches_by_frame[frame][0])):
            gt_cell = matches_by_frame[frame][0][idx]
            if gt_cell == -1:
                continue
            gt_pos = location_by_frame[frame][gt_cell]
            gt_class = track_info_by_frame[frame][gt_cell][-1]
            gt_cell_id = track_info_by_frame[frame][gt_cell][0]

            pred_cell_fr_idx = matches_by_frame[frame][1][idx]
            pred_cell_pos = pred_cells_fr[pred_cell_fr_idx]
            pred_probs = pred_cells_by_frame[frame][pred_cell_pos]
            pred_class = np.argmax(pred_probs)
            if pred_class != normal_class_id and \
               pred_probs[pred_class] < config.inference.prob_threshold:
                pred_class = normal_class_id

            dist = np.linalg.norm(
                np.array(gt_pos) -
                np.array(pred_cell_pos))

            if dist >= matching_threshold:
                logger.info("invalid match, distance too high: %s %s: %s",
                            gt_pos, pred_cell_pos, dist)
                results[gt_class]['fn'] += 1
                results[gt_class]['gt'] += 1
                results[pred_class]['fp'] += 1
                results[pred_class]['pred'] += 1
                invalid_dist_too_high_match_gt.append(
                    matches_by_frame[frame][0][idx])
                matches_by_frame[frame][0][idx] = -1
                matches_by_frame[frame][1][idx] = -1
                for midx in range(len(matches_by_frame[frame][0])):
                    gt_cell = matches_by_frame[frame][0][midx]
                    if gt_cell == -1:
                        continue
                    gt_pos = location_by_frame[frame][gt_cell]

                    pred_cell_fr_idx = matches_by_frame[frame][1][midx]
                    pred_cell_pos = pred_cells_fr[pred_cell_fr_idx]
                    logger.debug("frame: {}, gt: {}, pred {}".format(
                        frame, gt_pos, pred_cell_pos))
                continue
            else:
                num_matches += 1

                parent_id = gt_parent_map.get(tuple([l for l in gt_pos.ravel()]), None)
                if parent_id is None:
                    gt_idx = track_info_by_frame[frame][gt_cell][0]
                    logger.debug("id {} pos {}".format(gt_idx, gt_pos))
                if parent_id is not None:
                    parent_gt_pos = gt_idx_map[parent_id][0]
                    dist_parent = np.linalg.norm(
                        np.array(parent_gt_pos) -
                        np.array(pred_cell_pos))
                    if dist_parent >= 25:
                        logger.debug(("parent pos {}, pos {}, pred_cell pos {}, "
                                      "dist parent {}, dist {}").format(
                                          parent_gt_pos, gt_pos, pred_cell_pos,
                                          dist_parent, dist))

            # if set, store probable true div state of node candidate in db
            # (based on state of corresponding gt match)
            if set_gt_db:
                query = {"t": pred_cell_pos[0],
                         "z": pred_cell_pos[1],
                         "y": pred_cell_pos[2],
                         "x": pred_cell_pos[3]}
                update = {"$set": {"probable_gt_state": int(gt_class),
                                   "probable_gt_cell_id": gt_cell_id}}
                if not config.evaluate.dry_run:
                    logger.debug("updating %s with %s", query, update)
                    cells_db.update_one(query, update)
                else:
                    logger.debug("%s, %s", query, update)

            results[pred_class]['pred'] += 1
            results[gt_class]['gt'] += 1

            if gt_class == pred_class:
                results[gt_class]['tp'] += 1
                logger.debug("tp: gt {}/{} - pred {}/{}/{}".format(
                    gt_pos, gt_class, pred_cell_pos, pred_class, pred_probs))
            else:
                results[gt_class]['fn'] += 1
                results[pred_class]['fp'] += 1
                logger.info("fp/fn: gt {}/{} - pred {}/{}/{}".format(
                    gt_pos, gt_class, pred_cell_pos, pred_class, pred_probs))

            list_gt_class.append(gt_class)
            list_pred_class.append(pred_class)
            # the next two if/else are only for logging
            # and if fp divisions off by one frame should not be counted
            # TODO: one off fn divs?
            if gt_class == mother_class_id:
                if pred_class != mother_class_id:
                    logger.debug("mother fn %s %s",
                                 location_by_frame[frame][gt_cell],
                                 pred_cell_pos)
            else:
                if pred_class == mother_class_id:
                    loc = location_by_frame[frame][gt_cell].ravel()
                    parent_id = gt_parent_map.get(tuple([l for l in loc]), None)
                    children = gt_children_map.get(tuple([l for l in loc]), [])
                    is_fp = True
                    if config.evaluate.one_off:
                        # if there is a division one time frame before
                        if parent_id is not None and \
                           gt_idx_map[parent_id][1] == 1:
                            is_fp = False
                        # or one time frame after
                        elif children:
                            for child_id in children:
                                if gt_idx_map[child_id][1] == 1:
                                    is_fp = False
                    if is_fp:
                        logger.debug("mother fp %s %s",
                                     location_by_frame[frame][gt_cell],
                                     pred_cell_pos)
                    else:
                        # don't count fp
                        results[mother_class_id]['fp'] -= 1

            if gt_class == daughter_class_id:
                if pred_class != daughter_class_id:
                    logger.debug("daughter fn %s %s",
                                 location_by_frame[frame][gt_cell],
                                 pred_cell_pos)
            else:
                if pred_class == daughter_class_id:
                    loc = location_by_frame[frame][gt_cell].ravel()
                    parent_id = gt_parent_map.get(tuple([l for l in loc]), None)
                    children = gt_children_map.get(tuple([l for l in loc]), [])
                    is_fp = True
                    if config.evaluate.one_off:
                        # if there is a division one time frame before
                        if parent_id is not None and \
                           gt_idx_map[parent_id][1] == 2:
                            is_fp = False
                        # or one time frame after
                        elif children:
                            for child_id in children:
                                if gt_idx_map[child_id][1] == 2:
                                    is_fp = False
                    if is_fp:
                        logger.debug("daughter fp %s %s",
                                     location_by_frame[frame][gt_cell],
                                     pred_cell_pos)
                    else:
                        # don't count fp
                        results[daughter_class_id]['fp'] -= 1

        if num_matches - len(track_info_by_frame[frame]) != 0:
            logger.info(("frame: {} num matches: {} (num gt {}, num pred {}, "
                         "missing {})").format(
                             frame, num_matches,
                             len(track_info_by_frame[frame]), len(pred_cells_fr),
                             num_matches-len(track_info_by_frame[frame])))

        if num_matches - len(track_info_by_frame[frame]) < 0:
            for gID, gt_cell in enumerate(track_info_by_frame[frame]):
                if gID not in matches_by_frame[frame][0] and \
                   gID not in invalid_dist_too_high_match_gt:
                    gt_class = gt_cell[-1]
                    results[gt_class]['fn'] += 1
                    results[gt_class]['gt'] += 1
                    gt_pos = location_by_frame[frame][gID]
                    logger.info("fn: gt pos {}".format(gt_pos))
                    for midx in range(len(matches_by_frame[frame][0])):
                        gt_cell = matches_by_frame[frame][0][midx]
                        if gt_cell == -1:
                            continue
                        gt_pos = location_by_frame[frame][gt_cell]

                        pred_cell_fr_idx = matches_by_frame[frame][1][midx]
                        pred_cell_pos = pred_cells_fr[pred_cell_fr_idx]
                        logger.debug("frame: {}, gt: {}, pred {}".format(
                            frame, gt_pos, pred_cell_pos))

        if num_matches - len(pred_cells_fr) < 0:
            for pid, pred_cell in enumerate(pred_cells_fr):
                if pid not in matches_by_frame[frame][1]:
                    pred_probs = pred_cells_by_frame[frame][pred_cell]
                    pred_class = np.argmax(pred_probs)
                    results[pred_class]['fp'] += 1
                    results[pred_class]['pred'] += 1
                    logger.info("fp: pred pos {}".format(pred_cell))

    # Print the confusion matrix
    logger.info("Confusion Matrix: (pred across top, actual down size)")
    confusion_matrix = metrics.confusion_matrix(list_gt_class, list_pred_class)
    logger.info(confusion_matrix)

    logger.info("Classification Report:")
    # Print the precision and recall, among other metrics
    logger.info(metrics.classification_report(
        list_gt_class,
        list_pred_class,
        digits=3))

    return results


def compress_and_print_results(config, results):
    classes = list(config.model.classes)
    class_ids = list(config.model.class_ids)
    if config.model.with_polar:
        classes.append("polar")
        class_ids.append(3)

    for cls_idx, cls in zip(class_ids, classes):
        logger.info("{}   GT: {}".format(cls, results[cls_idx]['gt']))
        logger.info("{}   P : {}".format(cls, results[cls_idx]['pred']))
        logger.info("{}   TP: {}".format(cls, results[cls_idx]['tp']))
        logger.info("{}   FP: {}".format(cls, results[cls_idx]['fp']))
        logger.info("{}   FN: {}".format(cls, results[cls_idx]['fn']))
        logger.info("")
    logger.info("")

    for cls_idx, cls in zip(class_ids, classes):
        results[cls_idx]['TPR'] = \
            results[cls_idx]['tp']/max(1.0, results[cls_idx]['gt'])
        logger.info("{}   TPR: {}".format(cls, results[cls_idx]['TPR']))
        results[cls_idx]['FPR'] = \
            results[cls_idx]['fp']/max(1.0, results[cls_idx]['pred'])
        logger.info("{}   FPR: {}".format(cls, results[cls_idx]['FPR']))
        results[cls_idx]['FNR'] = \
            results[cls_idx]['fn']/max(1.0, results[cls_idx]['gt'])
        logger.info("{}   FNR: {}".format(cls, results[cls_idx]['FNR']))
        logger.info("")
    logger.info("")

    for cls_idx, cls in zip(class_ids, classes):
        results[cls_idx]['AP'] = (
            results[cls_idx]['tp']/max(1.0,
                                       results[cls_idx]['tp'] +
                                       results[cls_idx]['fp'] +
                                       results[cls_idx]['fn']))
        logger.info("{}   AP: {}".format(cls, results[cls_idx]['AP']))
    logger.info("(used threshold: %s)", config.inference.prob_threshold)
    logger.info("(sample: %s)", config.inference.data_source.datafile.filename)
    logger.info("")

    results_out = {}
    tpr, fpr, fnr, ap = 0, 0, 0, 0
    for cls_idx, cls in zip(class_ids, classes):
        results_out[cls] = {}
        tpr += results[cls_idx]['TPR']
        results_out[cls]['TPR'] = results[cls_idx]['TPR']
        fpr += results[cls_idx]['FPR']
        results_out[cls]['FPR'] = results[cls_idx]['FPR']
        fnr += results[cls_idx]['FNR']
        results_out[cls]['FNR'] = results[cls_idx]['FNR']
        ap += results[cls_idx]['AP']
        results_out[cls]['AP'] = results[cls_idx]['AP']

    results_out['mixed'] = {}
    cnt_cls = len(classes)
    results_out['mixed']['TPR'] = tpr / cnt_cls
    results_out['mixed']['mean_acc'] = tpr / cnt_cls
    results_out['mixed']['FPR'] = fpr / cnt_cls
    results_out['mixed']['FNR'] = fnr / cnt_cls
    results_out['mixed']['AP'] = ap / cnt_cls

    return results_out
