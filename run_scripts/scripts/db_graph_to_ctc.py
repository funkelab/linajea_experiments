import argparse
from copy import deepcopy
import logging
import math
import os
import time
import types

from natsort import natsorted
import networkx as nx
import numpy as np
import h5py
import toml
import zarr

import daisy
import linajea
import linajea.evaluation
import linajea.tracking
from linajea.visualization.ctc import write_ctc

logger = logging.getLogger(__name__)

logging.basicConfig(level=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--cls_config', type=str, default=None)
    parser.add_argument('--no_cck', dest="cck", action="store_false",
                        help='use cell cycle key/state classifier?')
    parser.add_argument('-o', '--out_dir', type=str)
    parser.add_argument('-p', '--param_id', type=int, default=None)
    parser.add_argument('-t', '--type', default=None,
                        choices=['ssvm',
                                 'best'])
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    parser.add_argument('--frame_begin', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=None)
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint/iteration to predict')
    parser.add_argument('-g', '--gt', action='store_true')
    parser.add_argument('--sphere', action='store_true')
    parser.add_argument('--ws', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--fg_threshold', type=float, default=0.5)
    parser.add_argument('--setup', type=str,
                        help='path to setup dir')
    parser.add_argument('--sample', type=str, default=None)
    args = parser.parse_args()

    print(args)
    if args.cls_config is not None and args.cck:
        with open(args.cls_config, 'r') as f:
            cls_config = toml.load(f)
    else:
        cls_config = None

    for config in linajea.getNextInferenceData(args):
        data = config.inference.data_source
        fn = data.datafile.filename

        if config.evaluate.parameters.roi is not None:
            assert config.evaluate.parameters.roi.shape[0] <= data.roi.shape[0], \
                "your evaluation ROI is larger than your data roi!"
            data.roi = config.evaluate.parameters.roi
        else:
            config.evaluate.parameters.roi = data.roi
        evaluate_roi = daisy.Roi(offset=data.roi.offset,
                                 shape=data.roi.shape)
        if args.frame_end is not None:
            tmp_roi = daisy.Roi(offset=(0, 0, 0, 0),
                                shape=(args.frame_end, None, None, None))
            evaluate_roi = evaluate_roi.intersect(tmp_roi)

        if args.gt:
            logger.info("db {}".format(data.gt_db_name))
            gt_db = linajea.CandidateDatabase(
                data.gt_db_name,
                config.general.db_host)
            graph = gt_db[evaluate_roi]

            if config.inference.data_source.gt_db_name_polar is not None and \
               not config.evaluate.parameters.filter_polar_bodies and \
               not config.evaluate.parameters.filter_polar_bodies_key:
                logger.info("polar bodies are not filtered, adding polar body GT..")
                gt_db_polar = linajea.CandidateDatabase(
                    data.gt_db_name_polar, config.general.db_host)
                gt_polar_subgraph = gt_db_polar[evaluate_roi]
                gt_mx_id = max(graph.nodes()) + 1
                mapping = {n: n+gt_mx_id for n in gt_polar_subgraph.nodes()}
                gt_polar_subgraph = nx.relabel_nodes(gt_polar_subgraph, mapping,
                                                     copy=False)

                graph.update(gt_polar_subgraph)
        else:
            logger.info("db {}".format(data.db_name))
            if args.param_id is not None:
                db = linajea.CandidateDatabase(
                    data.db_name,
                    config.general.db_host,
                    parameters_id=args.param_id)
            else:
                assert args.type is not None, "set param_id or type!"
                db = linajea.CandidateDatabase(
                    data.db_name,
                    config.general.db_host)
                ssvm_params = load_ssvm_params(args, config, cls_config=cls_config)
                if args.type == "ssvm":
                    params = ssvm_params
                    assert params is not None, \
                        "unable to load ssvm params"
                else: #if args.type == "best":
                    params = get_best_params_from_db(args, config,
                                                     cls_config=cls_config,
                                                     ssvm_params=ssvm_params)
                    assert params is not None, \
                        "unable to find valid/best params"
                set_params(config, params)
                param_id = db.get_parameters_id_round(
                    config.solve.parameters[0],
                    fail_if_not_exists=True)
                logger.info(f"selected param_id {param_id}")
                db.set_parameters_id(param_id)

            graph = db.get_selected_graph(evaluate_roi)

            if config.evaluate.parameters.filter_polar_bodies or \
               config.evaluate.parameters.filter_polar_bodies_key:
                if not config.evaluate.parameters.filter_polar_bodies and \
                   config.evaluate.parameters.filter_polar_bodies_key is not None:
                    pb_key = config.evaluate.parameters.filter_polar_bodies_key
                else:
                    pb_key = config.solve.parameters[0].cell_cycle_key + "polar"
                tmp_subgraph = db.get_selected_graph(evaluate_roi)
                for node in list(tmp_subgraph.nodes()):
                    if tmp_subgraph.degree(node) > 2:
                        es = list(tmp_subgraph.predecessors(node))
                        tmp_subgraph.remove_edge(es[0], node)
                        tmp_subgraph.remove_edge(es[1], node)
                rec_graph = linajea.tracking.TrackGraph(
                    tmp_subgraph, frame_key='t', roi=tmp_subgraph.roi)

                for track in rec_graph.get_tracks():
                    cnt_nodes = 0
                    cnt_polar = 0
                    for node_id, node in track.nodes(data=True):
                        cnt_nodes += 1
                        try:
                            if node[pb_key] > 0.5:
                                cnt_polar += 1
                        except KeyError:
                            pass
                    if cnt_polar/cnt_nodes > 0.5:
                        graph.remove_nodes_from(track.nodes())
                        logger.info("removing %s potential polar nodes", len(track.nodes()))

        logger.info("Read %d cells and %d edges",
                    graph.number_of_nodes(),
                    graph.number_of_edges())

        track_graph = linajea.tracking.TrackGraph(
            graph, frame_key='t', roi=graph.roi)

        if args.gt:
            txt_fn = "man_track.txt"
            tif_fn = "man_track{:03d}.tif"
        else:
            txt_fn = "res_track.txt"
            tif_fn = "mask{:03d}.tif"

        shape = None
        if os.path.isfile(os.path.join(fn, "data_config.toml")):
            data_config = linajea.load_config(os.path.join(fn, "data_config.toml"))
            shape = data_config['general']['shape']
        voxel_size = data.voxel_size

        if args.ws is not None and not args.gt:
            if args.ws == "surface":
                surf_fn = os.path.join(
                    config.predict.output_zarr_prefix,
                    os.path.basename(config.general.setup_dir),
                    os.path.basename(fn) + "predictions" + str(config.inference.checkpoint) +
                    "_" + str(config.inference.cell_score_threshold).replace(".", "_") + ".zarr")
            else:
                surf_fn = args.ws
            logger.info("file used for watershed surface: %s", surf_fn)
            surface = np.array(zarr.open(surf_fn, 'r')[
                    'volumes/cell_indicator'][:args.frame_end])
            if shape is None:
                shape = surface.shape
            if args.mask is not None:
                filename_mask = args.mask
            else:
                filename_mask = os.path.join(fn,
                                             data_config['general']['mask_file'])
            with h5py.File(filename_mask, 'r') as f:
                mask = np.array(f['volumes/mask'])
                mask = np.reshape(mask, (1, ) + mask.shape)
        else:
            surface = None
            mask = None

        if args.out_dir is None:
            out_dir = os.path.join(
                # "pbf",
                # "greedy",
                # "wo_cls2",
                "ssvm",
                # "w_cls",
                os.path.basename(config.general.setup_dir) + "_ctc_" + os.path.basename(fn),
                ("01_GT/TRA" if args.gt else "01_RES"))
        else:
            out_dir = args.out_dir
        logger.info("output directory: %s", out_dir)
        assert shape is not None, "TODO: set shape of sample"
        write_ctc(track_graph, args.frame_begin, args.frame_end, shape,
                  out_dir, txt_fn, tif_fn, voxel_size=voxel_size,
                  paint_sphere=args.sphere, gt=args.gt, surface=surface,
                  fg_threshold=args.fg_threshold, mask=mask)

        logger.info("stopping after first data sample")
        break

def load_ssvm_params(args, config, cls_config=None):
    logger.debug("running load ssvm")
    struct_svm_dir = "ssvm"
    # if not cls_config:
    #     struct_svm_dir += "_wo_cls"
    if cls_config is not None:
        struct_svm_dir += "_cls"
    struct_svm_dir += "_ckpt_{}".format(config.inference.checkpoint)
    if args.validation:
        fn = os.path.basename(config.test_data.data_sources[0].datafile.filename)
    else:
        fn = os.path.basename(config.validate_data.data_sources[0].datafile.filename)
    struct_svm_dir += "_" + fn
    struct_svm_dir = os.path.join(config.general.setup_dir,
                                  struct_svm_dir)
    # backwards compatibility
    if not os.path.isdir(struct_svm_dir) and "mskcc_e" in fn:
        struct_svm_dirT = struct_svm_dir[:-len(fn)]
        fn = fn.split("_")[-1]
        struct_svm_dirT += fn
        if os.path.isdir(struct_svm_dirT):
            struct_svm_dir = struct_svm_dirT

    ssvm_file = os.path.join(struct_svm_dir, "ssvm.txt")
    logger.info("checking %s for ssvm params", ssvm_file)
    if not os.path.isfile(ssvm_file):
        logger.info("file does not exist, skipping")
        return None
    with open(ssvm_file, 'r') as f:
        output = list(f)
        for line in output[::-1]:
            if "ε" in line and "INFO" in line and "is:" in line:
                eps = float(line.split()[-1])
                if abs(eps) > 1000:
                    logger.warning("unable to compute valid ssvm params")
                    return None
                else:
                    logger.info("found valid ssvm params in %s", struct_svm_dir)
                break
        params = {}
        params["weight_node_score"] =   float(output[-9])
        params["selection_constant"] =  float(output[-8])
        params["track_cost"] =          float(output[-7])
        # params["disappear_cost"] =     # = 0.0/not used
        params["weight_division"] =     float(output[-5])
        params["division_constant"] =   float(output[-4])
        params["weight_child"] =        float(output[-3])
        params["weight_continuation"] = float(output[-2])
        params["weight_edge_score"] =   float(output[-1])
        if cls_config is not None:
            params["cell_cycle_key"] = get_and_set_best_cls_params(
                cls_config, validation=args.validation)

        logger.debug("ssvm params: %s", params)
    return params


def set_params(config, params):
    config.solve.parameters[0].weight_node_score =   params["weight_node_score"]
    config.solve.parameters[0].selection_constant =  params["selection_constant"]
    config.solve.parameters[0].track_cost =          params["track_cost"]
    # config.solve.parameters[0].disappear_cost =    params["disappear_cost"] # = not used
    config.solve.parameters[0].weight_division =     params["weight_division"]
    config.solve.parameters[0].division_constant =   params["division_constant"]
    config.solve.parameters[0].weight_child =        params["weight_child"]
    config.solve.parameters[0].weight_continuation = params["weight_continuation"]
    config.solve.parameters[0].weight_edge_score =   params["weight_edge_score"]
    if 'cell_cycle_key' in params:
        config.solve.parameters[0].cell_cycle_key = params["cell_cycle_key"]


def dump_config(config):
    path = os.path.join(config["general"]["setup_dir"],
                        "tmp_configs",
                        "config_{}.toml".format(
                            time.time()))
    logger.debug("config dump path: %s", path)
    with open(path, 'w') as f:
        toml.dump(config, f)
    return path


def get_best_params_from_db(args, config, cls_config=None,
                            ssvm_params=None):
    if cls_config is not None:
        cck = get_and_set_best_cls_params(cls_config, validation=args.validation)

    logger.debug("running get params from db")

    tmp_args = types.SimpleNamespace(
        config=config.path, validation=not args.validation,
        validate_on_train=False, checkpoint=0, val_param_id=None,
        param_id=None, param_ids=None, param_list_idx=None)
    for inf_config in linajea.getNextInferenceData(tmp_args, is_evaluate=True):
        res = linajea.evaluation.get_results_sorted(
            inf_config,
            filter_params={"val": True},
            score_columns=['fp_edges', 'fn_edges',
                           'identity_switches',
                           'fp_divisions', 'fn_divisions'],
            sort_by="sum_errors")
        break
    assert not res.empty, "no results found in db"
    params = None
    for entry in res.to_dict(orient="records"):
        found = False
        if cls_config is not None:
            if entry.get('cell_cycle_key') == cck:
                found = True
        else:
            if 'cell_cycle_key' not in entry or \
               not entry['cell_cycle_key'] or \
               (not isinstance(entry["cell_cycle_key"], str) and \
                math.isnan(entry["cell_cycle_key"])):
                found = True
        if found:
            params = {}
            params["weight_node_score"] =   entry["weight_node_score"]
            params["selection_constant"] =  entry["selection_constant"]
            params["track_cost"] =          entry["track_cost"]
            # params["disappear_cost"] =    entry["disappear_cost"] # = 0.0/not used
            params["weight_division"] =     entry["weight_division"]
            params["division_constant"] =   entry["division_constant"]
            params["weight_child"] =        entry["weight_child"]
            params["weight_continuation"] = entry["weight_continuation"]
            params["weight_edge_score"] =   entry["weight_edge_score"]
            if "cell_cycle_key" in entry:
                params["cell_cycle_key"] =      entry["cell_cycle_key"]
                if not isinstance(params["cell_cycle_key"], str) and \
                   math.isnan(params["cell_cycle_key"]):
                    params["cell_cycle_key"] = None

            if params != ssvm_params:
                break
    else:
        raise RuntimeError("no result with correct cck found")

    logger.info("best entry from db: %s: %s", entry["_id"], params)

    return params


def get_and_set_best_cls_params(cls_config, validation,
                                all_th=False):
    logger.debug("running get cls params")
    metrics = {}
    if validation:
        dss = cls_config['test_data']['data_sources']
        roi = cls_config['test_data']['roi']
    else:
        dss = cls_config['validate_data']['data_sources']
        roi = cls_config['validate_data']['roi']

    res_dir = os.path.join(cls_config["general"]["setup_dir"], "val")
    for chkpt_dir in natsorted(os.listdir(res_dir)):
        if "event" in chkpt_dir:
            continue
        chkpt = int(chkpt_dir)
        for th_dir in natsorted(os.listdir(os.path.join(res_dir, chkpt_dir))):
            th = float(th_dir.split("prob_threshold_")[-1].replace("_", "."))
            if not all_th and th != 0.1:
                continue
            for kind in os.listdir(os.path.join(res_dir, chkpt_dir, th_dir)):
                for sample in dss:
                    sample_fn = os.path.basename(sample['datafile']['filename'])
                    if "polar" in sample_fn or (sample_fn + "_") not in kind:
                        continue

                    if roi['shape'][0] != \
                       int(kind.split("roi")[-1].split("_")[0]):
                        continue

                    swa = "swa" in kind
                    if "ttr" in kind:
                        ttr = int(kind.split("ttr")[-1].split("_")[0])
                    else:
                        ttr = 1

                    with open(os.path.join(res_dir, chkpt_dir,
                                           th_dir, kind), 'r') as f:
                        results = toml.load(f)
                    params = (sample_fn, chkpt, th,
                              "w_swa" if swa else "wo_swa",
                              "ttr {}".format(ttr))
                    metric = results['mixed']['AP']
                    metrics[params] = metric
                    logger.info("{:.7f} {}".format(metric, params))

    if len(metrics) == 0:
        raise RuntimeError("get_best_cls_params: nothing found")

    best_params = max(metrics, key=metrics.get)
    best_metrics = metrics[best_params]

    logger.debug("best: {} {}".format(best_metrics, best_params))
    sample, checkpoint, th, swa, ttr = best_params
    cls_config['test_data']['checkpoint'] = checkpoint
    cls_config['test_data']['prob_threshold'] = th
    cls_config['predict']['use_swa'] = swa == "w_swa"
    cls_config['predict']['test_time_reps'] = int(ttr.split()[-1])

    cck = os.path.join(
        os.path.basename(cls_config["general"]["setup_dir"]).split("classifier_")[-1],
        "test",
        str(checkpoint) + "_")
    logger.debug("compute cell_cycle_key: %s", cck)

    return cck


if __name__ == "__main__":
    main()
