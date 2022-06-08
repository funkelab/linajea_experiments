import argparse
from copy import deepcopy
import glob
import logging
import math
import os
import subprocess
import sys
import time
import types

import attr
from natsort import natsorted
import toml

from linajea import (CandidateDatabase,
                     checkOrCreateDB,
                     getNextInferenceData,
                     load_config)
from linajea.config import (# TrackingConfig,
                            # CellCycleConfig,
                            SolveParametersMinimalConfig,
                            maybe_fix_config_paths_to_machine_and_load)
import linajea.evaluation

logger = logging.getLogger(__name__)
logging.basicConfig(level=10)

def do_predict_eval_ckpts_cls_val(args, cls_config, track_config):
    logger.debug("running predict eval cls")
    # bsub -I -n 4 -q gpu_tesla -gpu num=1:mps=no
    # python run_lcdc.py -c experiments/classifier_emb1_swa/config.toml
    # -id experiments/classifier_emb1_swa -d validate_checkpoints
    cls_config["validate_data"]["use_database"] = False
    path = dump_config(cls_config)
    cmd = ["bsub", "-I", "-n", "4", "-q", "gpu_tesla", "-gpu", "num=1:mps=no",
           # "-r\"rusage[mem=25600]\"",
           "python", "run_lcdc.py",
           "-c", path, "-id", cls_config["general"]["setup_dir"],
           "-d", "validate_checkpoints"]
    run_cmd(args, cmd)


def do_predict_cls(args, cls_config, validation):
    logger.debug("running predict cls")
    # bsub -I -n 4 -q gpu_tesla -gpu num=1:mps=no
    # python run_lcdc.py -c experiments/classifier_emb1_swa/config.toml
    # -id experiments/classifier_emb1_swa -d predict [--swap_val_test]
    if validation:
        cls_config = swap_val_test(cls_config)
    cls_config["test_data"]["use_database"] = True
    path = dump_config(cls_config)

    cmd = ["bsub", "-I", "-n", "4", "-q", "gpu_tesla", "-gpu", "num=1:mps=no",
           "python", "run_lcdc.py",
           "-c", path, "-id", cls_config["general"]["setup_dir"],
           "-d", "predict"]

    run_cmd(args, cmd)

    if validation:
        cls_config = swap_val_test(cls_config)
    cls_config["test_data"]["use_database"] = False


def do_predict_track(args, track_config, validation):
    logger.debug("running predict track")
    # python ~/linajea_experiments/unet_setups/celegans_setups/run_celegans.py
    # --config /path/to/config/config_celegans.toml
    # --predict  --validation --interactive
    path = dump_config(track_config)
    cmd = ["python", "run_celegans.py",
           "--config", path, "--predict"]#, "--interactive"]
    if validation:
        cmd.append("--validation")
    run_cmd(args, cmd)


def do_extract_track(args, track_config, validation):
    logger.debug("running extract track")
    # python ~/linajea_experiments/unet_setups/celegans_setups/run_celegans.py
    # --config /path/to/config/config_celegans.toml
    # --extract  --validation --interactive
    path = dump_config(track_config)
    cmd = ["python", "run_celegans.py",
           "--config", path, "--extract", "--interactive"]
    if validation:
        cmd.append("--validation")
    run_cmd(args, cmd)


def do_solve_track(args, track_config, cls, search, validation, ssvm=False):
    logger.debug("running solve track")
    # python ~/linajea_experiments/unet_setups/celegans_setups/run_celegans.py
    # --config /path/to/config/config_celegans.toml
    # --solve  --validation --interactive
    solve_parameters = deepcopy(track_config['solve']['parameters'])

    if not validation:
        assert not search, "there should be no search on test data"

        if track_config['solve'].get('greedy', False):
            track_config['solve']['parameters'] = [set_greedy_params(
                track_config, cls)]
        elif not ssvm:
            track_config['solve']['parameters'] = [get_best_params_from_db(
                track_config, cls, validation=True)]
        else:
            # using ssvm parameters, already set
            pass

        if cls:
            ps = track_config['solve']['parameters'][0]
            assert ('cell_cycle_key' in ps and
                    ps['cell_cycle_key'] not in ([], '', [''])), \
                    "no cell_cycle_key in solve parameters!"
        else:
            if 'cell_cycle_key' in track_config['solve']['parameters'][0]:
                del track_config['solve']['parameters'][0]['cell_cycle_key']

    elif search:
        assert ('parameters_search_grid' in track_config['solve'] or
                'parameters_search_random' in track_config['solve']), \
            "no solve search parameters in config!"

        if cls:
            if 'parameters_search_grid' in track_config['solve']:
                ps = track_config['solve']['parameters_search_grid']
                assert ('cell_cycle_key' in ps and
                        ps['cell_cycle_key'] not in ([], '', [''])), \
                        "no cell_cycle_key in solve search grid parameters!"
            if 'parameters_search_random' in track_config['solve']:
                ps = track_config['solve']['parameters_search_random']
                assert ('cell_cycle_key' in ps and
                        ps['cell_cycle_key'] not in ([], '', [''])), \
                        "no cell_cycle_key in solve search random parameters!"
        else:
            if 'parameters_search_grid' in track_config['solve']:
                if 'cell_cycle_key' in track_config['solve']['parameters_search_grid']:
                    del track_config['solve']['parameters_search_grid'][
                        'cell_cycle_key']
            if 'parameters_search_random' in track_config['solve']:
                if 'cell_cycle_key' in track_config['solve']['parameters_search_random']:
                    del track_config['solve']['parameters_search_random'][
                        'cell_cycle_key']
    else:
        if not ssvm:
            raise RuntimeError("invalid args")

    cmd = ["python", "run_celegans.py",
           "--config", "REPLACE", "--solve", "--interactive"]
    if validation:
        cmd.append("--validation")

    if search and 'parameters_search_grid' in track_config['solve']:
        logger.debug("solving grid_search")
        track_config['solve']['grid_search'] = True
        path = dump_config(track_config)
        cmd[3] = path
        run_cmd(args, cmd)
        track_config['solve']['grid_search'] = False
    if search and 'parameters_search_random' in track_config['solve']:
        logger.debug("solving random_search")
        track_config['solve']['random_search'] = True
        path = dump_config(track_config)
        cmd[3] = path
        run_cmd(args, cmd)
        track_config['solve']['random_search'] = False
    if not search:
        track_config['solve']['grid_search'] = False
        track_config['solve']['random_search'] = False
        path = dump_config(track_config)
        cmd[3] = path
        run_cmd(args, cmd)

    track_config['solve']['grid_search'] = False
    track_config['solve']['random_search'] = False
    track_config['solve']['parameters'] = solve_parameters


def do_solve_write_ssvm(args, track_config, cls):
    logger.debug("running write ssvm")
    track_config['solve']['write_struct_svm'] = "ssvm"
    if not cls:
        track_config['solve']['write_struct_svm'] += "_wo_cls"
    do_solve_track(args, track_config, cls=cls, search=False, validation=True,
                   ssvm=True)
    del track_config['solve']['write_struct_svm']


def do_solve_compute_ssvm(args, track_config, cls, local=False):
    logger.debug("running compute ssvm")
    if local:
        cmd_pref = []
    else:
        cmd_pref = ["bsub", "-I", "-n", "1"]

    path = dump_config(track_config)
    cmd = cmd_pref + ["python", "../../run_scripts/set_probable_gt_state.py",
           "--config", path, "--validation"]
    run_cmd(args, cmd)

    cmd = cmd_pref + ["python", "../../run_scripts/add_best_effort_to_db.py",
           "--config", path, "--validation"]
    run_cmd(args, cmd)

    struct_svm_dir =  get_struct_svm_dir(track_config, cls)
    cmd = cmd_pref + ["python", "../../run_scripts/aggregate_struct_svm_data.py",
           "--config", path,
           "--dir", struct_svm_dir, "--validation"]
    run_cmd(args, cmd)

    os.chdir(struct_svm_dir)
    if not local:
        cmd_pref = ["bsub", "-K", "-n", "1"]
    cmd = cmd_pref + ["python", "../../../../../run_scripts/run_struct_svm.py"]
           # ">", "ssvm.txt", "2>&1"]
    with open("ssvm.txt", 'w') as of:
        run_cmd(args, cmd, stdout=of, stderr=subprocess.STDOUT)
    os.chdir(os.path.join(paths_machine['HOME'], "unet_setups",
                          "celegans_setups"))


def do_eval_track(args, track_config, cls, search, validation, frames=None,
                  ssvm=False):
    logger.debug("running eval track")
    # python ~/linajea_experiments/unet_setups/celegans_setups/run_celegans.py
    # --config /path/to/config/config_celegans.toml
    # --evaluate  --validation --eval_array_job
    solve_parameters = deepcopy(track_config['solve']['parameters'])
    eval_parameters = deepcopy(track_config['evaluate']['parameters'])

    if not validation:
        assert not search, "there should be no search on test data"
        if track_config['solve'].get('greedy', False):
            track_config['solve']['parameters'] = [set_greedy_params(
                track_config, cls)]
        elif not ssvm:
            track_config['solve']['parameters'] = [get_best_params_from_db(
                track_config, cls, validation=True)]
        else:
            # using ssvm parameters, already set
            pass
        # assume params in solve.parameters are correct or handled later
        if cls:
            ps = track_config['solve']['parameters'][0]
            assert ('cell_cycle_key' in ps and
                    ps['cell_cycle_key'] not in ([], '', [''])), \
                    "no cell_cycle_key in solve parameters!"
        else:
            if 'cell_cycle_key' in track_config['solve']['parameters'][0]:
                del track_config['solve']['parameters'][0]['cell_cycle_key']
    elif search:
        # handled later
        pass
    else:
        raise RuntimeError("invalid args")

    if frames is not None:
        if 'roi' not in eval_parameters:
            if validation:
                track_config['evaluate']['parameters']['roi'] = \
                    deepcopy(track_config['validate_data']['roi'])
            else:
                track_config['evaluate']['parameters']['roi'] = \
                    deepcopy(track_config['test_data']['roi'])
        track_config['evaluate']['parameters']['roi']['shape'][0] = frames

    track_config['solve']['grid_search'] = False
    track_config['solve']['random_search'] = False
    path = dump_config(track_config)
    cmd = ["python", "run_celegans.py",
           "--config", path, "--evaluate"]
    if validation:
        cmd.append("--validation")
    if search:
        cmd.append("--eval_array_job")
    else:
        cmd.append("--interactive")

    tmp_args = types.SimpleNamespace(
        config=path, validation=validation, validate_on_train=False,
        checkpoint=-1, val_param_id=None, param_id=None,
        param_ids=None, param_list_idx=None)
    results = []
    if validation:
        data_sources = deepcopy(track_config['validate_data']['data_sources'])
    else:
        data_sources = deepcopy(track_config['test_data']['data_sources'])
    for idx, inf_config in enumerate(getNextInferenceData(tmp_args,
                                                          is_evaluate=True)):
        if search:
            db = CandidateDatabase(inf_config.inference.data_source.db_name,
                                   inf_config.general.db_host)
            params_collection = db.database['parameters']
            max_param_id = params_collection.find_one({'_id': 'parameters'})['id']
            cmd.extend(["--param_ids", "1", str(max_param_id)])

        assert inf_config.inference.data_source.datafile.filename == \
                data_sources[idx]['datafile']['filename']
        if validation:
            track_config['validate_data']['data_sources'] = [data_sources[idx]]
        else:
            track_config['test_data']['data_sources'] = [data_sources[idx]]
        _ = dump_config(track_config)
        run_cmd(args, cmd)

        if not search:
            params = deepcopy(inf_config.solve.parameters[0])
            params.val = False
            inf_config.evaluate.parameters.filter_polar_bodies = False
            # TODO: aggregate over checkpoints/samples
            results.append(linajea.evaluation.get_result_params(
                inf_config,
                params))
    if validation:
        track_config['validate_data']['data_sources'] = data_sources
    else:
        track_config['test_data']['data_sources'] = data_sources

    track_config['solve']['parameters'] = solve_parameters
    if frames is not None:
        if 'roi' in eval_parameters:
            track_config['evaluate']['parameters'] = eval_parameters
        else:
            del track_config['evaluate']['parameters']['roi']

    return results


def do_solve_greedy(args, track_config):
    logger.debug("running solve greedy")
    track_config['solve']['greedy'] = True
    do_solve_track(args, track_config, cls=False, search=False, validation=False)
    track_config['solve']['greedy'] = False


def do_eval_greedy(args, track_config, frames=None):
    logger.debug("running eval greedy")
    track_config['solve']['greedy'] = True
    results = do_eval_track(args, track_config, cls=False, search=False,
                            validation=False, frames=frames)
    track_config['solve']['greedy'] = False
    return results


def do_solve_ssvm(args, track_config, cls):
    logger.debug("running solve ssvm")
    solve_parameters = deepcopy(track_config['solve']['parameters'])
    struct_svm_dir = get_struct_svm_dir(track_config, cls)
    success = load_ssvm_params(args, track_config, struct_svm_dir, cls)
    if not success:
        return False
    do_solve_track(args, track_config, cls=cls, search=False, validation=False,
                   ssvm=True)
    track_config['solve']['parameters'] = solve_parameters
    return True


def do_eval_ssvm(args, track_config, cls, frames=None):
    logger.debug("running eval ssvm")
    solve_parameters = deepcopy(track_config['solve']['parameters'])
    struct_svm_dir = get_struct_svm_dir(track_config, cls)
    success = load_ssvm_params(args, track_config, struct_svm_dir, cls)
    if not success:
        return False
    results = do_eval_track(args, track_config, cls=cls, search=False,
                            validation=False, frames=frames, ssvm=True)
    track_config['solve']['parameters'] = solve_parameters
    logger.info("ssvm res %s", results)
    return results


def run_cmd(args, cmd, shell=False, stdout=None, stderr=None):
    logger.info("cwd: {}".format(os.getcwd()))
    if args.local:
        cmd = cmd[(cmd.index("python")):]
    if args.dry_run:
        logger.info("would run: {}".format(' '.join(cmd)))
    else:
        logger.info("running: {}".format(' '.join(cmd)))
        _ = subprocess.run(cmd, check=True, shell=shell,
                           stdout=stdout, stderr=stderr)
        logger.info("done running: {}".format(' '.join(cmd)))


def swap_val_test(config):
    logger.info("swapping validation and test data!")
    config["test_data"]["data_sources"], \
        config["validate_data"]["data_sources"] = \
            config["validate_data"]["data_sources"], \
            config["test_data"]["data_sources"]

    return config


def get_struct_svm_dir(track_config, cls):
    struct_svm_dir = "ssvm"
    if not cls:
        struct_svm_dir += "_wo_cls"
    # if cls:
    #     struct_svm_dir += "_cls"
    struct_svm_dir += "_ckpt_{}".format(
        track_config['validate_data']['checkpoints'][0])
    fn = os.path.basename(
        track_config["validate_data"]["data_sources"][0]["datafile"]["filename"])
    if len(track_config["validate_data"]["data_sources"]) > 1:
        logger.warning(
            "computing SSVM only for first dataset! %s", fn)
    struct_svm_dir += "_" + fn
    struct_svm_dir = os.path.join(track_config["general"]["setup_dir"],
                                  struct_svm_dir)
    # backwards compatibility
    if not os.path.isdir(struct_svm_dir) and "mskcc_e" in fn:
        struct_svm_dirT = struct_svm_dir[:-len(fn)]
        fn = fn.split("_")[-1]
        struct_svm_dirT += fn
        if os.path.isdir(struct_svm_dirT):
            struct_svm_dir = struct_svm_dirT

    return struct_svm_dir


def load_ssvm_params(args, track_config, struct_svm_dir, cls):
    logger.debug("running load ssvm, dir: %s", struct_svm_dir)
    params = {}
    if args.ssvm_type is None or args.ssvm_type == "default":
        with open(os.path.join(struct_svm_dir, "ssvm.txt"), 'r') as f:
            output = list(f)
        for line in output[::-1]:
            if "ε" in line and "INFO" in line and "is:" in line:
                eps = float(line.split()[-1])
                if abs(eps) > 1000:
                    logger.warning("unable to compute valid ssvm params")
                    return False
                else:
                    logger.info("found valid ssvm params in dir %s", struct_svm_dir)
                break
    else:
        fls = glob.glob(os.path.join(struct_svm_dir, "logs/*.out"))
        logger.debug("checkings fls %s for ssvm type %s", fls, args.ssvm_type)
        for fl in fls:
            with open(os.path.join(struct_svm_dir, fl), 'r') as f:
                ln = next(f)
                if ln[:-1].split(" ")[-1] != args.ssvm_type:
                    continue
                output = list(f)
                for idx, ln in enumerate(output):
                    if "Subject: Job" in ln:
                        break
                if "Done" not in output[idx]:
                    continue
                logger.info("found valid ssvm params in file %s", os.path.join(struct_svm_dir, fl))
                output = output[idx-12:idx-3]
                break
        else:
            logger.warning("unable to compute valid ssvm params")
            return False

    params["weight_node_score"] =   float(output[-9])
    params["selection_constant"] =  float(output[-8])
    params["track_cost"] =          float(output[-7])
    # params["disappear_cost"] =      float(line[-6]) # = 0.0/not used
    params["weight_division"] =     float(output[-5])
    params["division_constant"] =   float(output[-4])
    params["weight_child"] =        float(output[-3])
    params["weight_continuation"] = float(output[-2])
    params["weight_edge_score"] =   float(output[-1])
    params["block_size"] = deepcopy(
        track_config['solve']['parameters'][0]['block_size'])
    params["context"] = deepcopy(
        track_config['solve']['parameters'][0]['context'])
    if cls:
        params["cell_cycle_key"] = track_config['solve']['parameters'][0][
            'cell_cycle_key']

    track_config['solve']['parameters'] = [params]
    logger.debug("ssvm params: %s", params)
    return True


def set_greedy_params(track_config, cls):
    assert not cls, "no classifier for greedy solving"
    params = {}
    params["weight_node_score"] =   0
    params["selection_constant"] =  0
    params["track_cost"] =          0
    # params["disappear_cost"] =    0 # = 0.0/not used
    params["weight_division"] =     0
    params["division_constant"] =   0
    params["weight_child"] =        0
    params["weight_continuation"] = 0
    params["weight_edge_score"] =   0
    params["block_size"] = deepcopy(
        track_config['solve']['parameters'][0]['block_size'])
    params["context"] = deepcopy(
        track_config['solve']['parameters'][0]['context'])

    return params


def get_best_params_from_db(track_config, cls, validation):
    logger.debug("running get params from db")
    tmp_path = dump_config(track_config)
    tmp_args = types.SimpleNamespace(
        config=tmp_path, validation=validation, validate_on_train=False,
        checkpoint=0, val_param_id=None, param_id=None,
        param_ids=None, param_list_idx=None)
    for inf_config in getNextInferenceData(tmp_args, is_evaluate=True):
        res = linajea.evaluation.get_results_sorted(
            inf_config,
            filter_params={"val": True},
            score_columns=['fp_edges', 'fn_edges',
                           'identity_switches',
                           'fp_divisions', 'fn_divisions'],
            sort_by="sum_errors")
        break
    assert not res.empty, "no results found in db"
    for entry in res.to_dict(orient="records"):
        if cls:
            if entry.get('cell_cycle_key'):
                break
        else:
            if 'cell_cycle_key' not in entry or \
               not entry['cell_cycle_key'] or \
               (not isinstance(entry["cell_cycle_key"], str) and \
                math.isnan(entry["cell_cycle_key"])):
                break
    else:
        raise RuntimeError("no result with correct cck found")
    logger.debug("best entry from db: %s", filter_results({0: [entry]}))
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
    params["cell_cycle_key"] =      entry.get('cell_cycle_key')
    if not isinstance(params["cell_cycle_key"], str) and \
       math.isnan(params["cell_cycle_key"]):
        params["cell_cycle_key"] = None
    params["block_size"] = deepcopy(
        track_config['solve']['parameters'][0]['block_size'])
    params["context"] = deepcopy(
        track_config['solve']['parameters'][0]['context'])

    logger.debug("best params from db: %s", params)
    return params


def get_and_set_best_cls_params(cls_config, track_config, validation,
                                all_th=False):
    logger.debug("running get cls params")
    metrics = {}
    if validation:
        suffix = "val"
    else:
        suffix = "test"
    res_dir = os.path.join(cls_config["general"]["setup_dir"], suffix)
    for chkpt_dir in natsorted(os.listdir(res_dir)):
        if "event" in chkpt_dir:
            continue
        chkpt = int(chkpt_dir)
        for th_dir in natsorted(os.listdir(os.path.join(res_dir, chkpt_dir))):
            th = float(th_dir.split("prob_threshold_")[-1].replace("_", "."))
            if not all_th and th != 0.1:
                continue
            for kind in os.listdir(os.path.join(res_dir, chkpt_dir, th_dir)):
                for sample in cls_config['validate_data']['data_sources']:
                    sample_fn = os.path.basename(sample['datafile']['filename'])
                    if "polar" in sample_fn or (sample_fn + "_") not in kind:
                        continue

                    if cls_config['validate_data']['roi']['shape'][0] != \
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
                    logger.debug("{:.7f} {}".format(metric, params))

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
    for i in range(len(track_config['solve'].get('parameters', []))):
        track_config['solve']['parameters'][i]['cell_cycle_key'] = cck
    if 'parameters_search_grid' in track_config['solve']:
        track_config['solve']['parameters_search_grid']['cell_cycle_key'] = [cck]
    if 'parameters_search_random' in track_config['solve']:
        track_config['solve']['parameters_search_random']['cell_cycle_key'] = [cck]


def filter_results(results):
    for _, resT in results.items():
        for res in resT:
            if res is None:
                continue
            del res['fn_edge_list']
            del res['fp_edge_list']
            del res['identity_switch_gt_nodes']
            del res['fp_div_rec_nodes']
            del res['no_connection_gt_nodes']
            del res['unconnected_child_gt_nodes']
            del res['unconnected_parent_gt_nodes']
            del res['tp_div_gt_nodes']

    return results


def dump_config(config):
    path = os.path.join(config["general"]["setup_dir"],
                        "tmp_configs",
                        "config_{}.toml".format(
                            time.time()))
    logger.debug("config dump path: %s", path)
    with open(path, 'w') as f:
        toml.dump(config, f)
    return path


if __name__ == "__main__":
    os.environ['GRB_LICENSE_FILE'] = "/misc/local/gurobi-9.1.2/gurobi.lic"

    logger.info("args: %s", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_config', type=str,
                        help='path to config file')
    parser.add_argument('--cls_config', type=str,
                        help='path to config file')
    parser.add_argument("--dry_run", action="store_true",
                        help="just print steps to run")
    parser.add_argument("--skip_predict_tracking", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_solve", action="store_true")
    parser.add_argument("--skip_solve_test", action="store_true")
    parser.add_argument("--skip_predict_cls", action="store_true")
    parser.add_argument("--skip_search", action="store_true")
    parser.add_argument("--skip_write_ssvm", action="store_true")
    parser.add_argument("--skip_ssvm", action="store_true")
    parser.add_argument("--compute_ssvm", action="store_true")
    parser.add_argument("--only_predict_test", action="store_true")
    parser.add_argument("--only_write_ssvm", action="store_true")
    parser.add_argument("--only_compute_ssvm", action="store_true")
    parser.add_argument("--only_greedy", action="store_true")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--swap_val_test", action="store_true",
                        help='swap val and test data')
    parser.add_argument('--frames', default=[], nargs='+', type=int,
                        help='Eval on X first frames, accepts multiple values')
    parser.add_argument('--ssvm_type', type=str, default=None,
                        help='which ssvm setup has been used to compute hyperparameters? (usually default is fine)')

    args = parser.parse_args()
    track_config = maybe_fix_config_paths_to_machine_and_load(
        args.track_config)
    # track_config = TrackingConfig(**track_config)
    os.makedirs(os.path.join(track_config["general"]["setup_dir"],
                             "tmp_configs"), exist_ok=True)
    if args.swap_val_test:
        track_config = swap_val_test(track_config)

    cls_config = maybe_fix_config_paths_to_machine_and_load(
        args.cls_config)
    # cls_config = CellCycleConfig(cls_config)
    os.makedirs(os.path.join(cls_config["general"]["setup_dir"],
                             "tmp_configs"), exist_ok=True)
    if args.swap_val_test:
        cls_config = swap_val_test(cls_config)
    cls_config['validate_data']['db_meta_info']['setup_dir'] = os.path.basename(
        track_config["general"]["setup_dir"])
    cls_config['test_data']['db_meta_info']['setup_dir'] = os.path.basename(
        track_config["general"]["setup_dir"])

    if args.frames:
        logger.debug("frames to eval: %s", args.frames)
        frames = args.frames
    else:
        if 'roi' in track_config['evaluate']['parameters']:
            frames = [track_config['evaluate']['parameters']['roi']['shape'][0]]
        else:
            frames = [track_config['test_data']['roi']['shape'][0]]

    paths_machine = load_config(os.path.join(os.environ['HOME'],
                                             "linajea_paths.toml"))

    environ_path = ["/misc/local/git/bin:",
                    "/misc/lsf/10.1/linux3.10-glibc2.17-x86_64/etc:",
                    "/misc/lsf/10.1/linux3.10-glibc2.17-x86_64/bin:",
                    "/groups/funke/home/hirschp/anaconda3/bin:",
                    "/groups/funke/home/hirschp/anaconda3/bin:",
                    "/groups/funke/home/hirschp/anaconda3/condabin:",
                    "/usr/share/Modules/bin:"
                    "/usr/local/bin:",
                    "/usr/bin:",
                    "/usr/local/sbin:"
                    "/usr/sbin:",
                    "/groups/funke/home/hirschp/.local/bin:",
                    "/groups/funke/home/hirschp/bin:"]
    environ_lib_path = ["/groups/funke/home/hirschp/anaconda3/lib:",
                        "/misc/local/gurobi-9.1.2/lib:"]

    environ_path_tf = ("/groups/funke/home/hirschp/anaconda3/envs/linajea_cudnn8/bin:" +
                       "".join(environ_path))
    environ_lib_path_tf = (
        "/groups/funke/home/hirschp/cudnn8:"
        "/misc/local/cuda-10.2/lib64:"
        "".join(environ_lib_path))

    environ_path_torch = ("/groups/funke/home/hirschp/anaconda3/envs/linajea_torch/bin:" +
                          "".join(environ_path))
    environ_lib_path_torch= (
        "/groups/funke/home/hirschp/cudnn-11.1-linux-x64-v8.0.5.39/cuda/lib64:"
        "".join(environ_lib_path))

    os.environ['LD_LIBRARY_PATH'] = environ_lib_path_torch
    os.environ['PATH'] = environ_path_torch
    os.chdir(os.path.join(paths_machine['HOME'], "unet_setups",
                          "celegans_setups"))

    results = {}

    if args.only_predict_test:
        do_predict_track(args, track_config, validation=False)
        exit()
    if args.only_write_ssvm:
        get_and_set_best_cls_params(cls_config, track_config, validation=True,
                                    all_th=False)
        do_solve_write_ssvm(args, track_config, cls=True)
        exit()
    if args.only_compute_ssvm:
        do_solve_compute_ssvm(args, track_config, cls=True, local=args.local)
        exit()

    if args.only_greedy:
        # . solve greedy test
        if not args.skip_solve:
            do_solve_greedy(args, track_config)

        # . evaluate greedy test
        for fr in frames:
            results.setdefault("greedy", {})[fr] = \
                do_eval_greedy(args, track_config, frames=fr)
        logger.info("greedy results: %s", filter_results(results["greedy"]))
        exit()

    # CLASSIFIER
    os.chdir(os.path.join(paths_machine['HOME'],
                          "classifier_setups", "celegans_setups"))

    os.environ['LD_LIBRARY_PATH'] = environ_lib_path_tf
    os.environ['PATH'] = environ_path_tf
    # . predict/evaluate checkpoints classifier validation
    if not args.skip_predict_cls:
        do_predict_eval_ckpts_cls_val(args, cls_config, track_config)

    get_and_set_best_cls_params(cls_config, track_config, validation=True,
                                all_th=False)

    ## TRACKING
    os.environ['LD_LIBRARY_PATH'] = environ_lib_path_torch
    os.environ['PATH'] = environ_path_torch
    os.chdir(os.path.join(paths_machine['HOME'], "unet_setups",
                          "celegans_setups"))
    # . predict tracking validation
    if not args.skip_predict_tracking:
        do_predict_track(args, track_config, validation=True)

    # . extract edges tracking validation
    if not args.skip_extract:
        do_extract_track(args, track_config, validation=True)

    # . predict classifier database validation
    if not args.skip_predict_cls:
        os.chdir(os.path.join(paths_machine['HOME'], "classifier_setups",
                              "celegans_setups"))
        os.environ['LD_LIBRARY_PATH'] = environ_lib_path_tf
        os.environ['PATH'] = environ_path_tf
        do_predict_cls(args, cls_config, validation=True)

        os.chdir(os.path.join(paths_machine['HOME'], "unet_setups",
                              "celegans_setups"))

        os.environ['LD_LIBRARY_PATH'] = environ_lib_path_torch
        os.environ['PATH'] = environ_path_torch

    # . solve with classifier search validation
    if not args.skip_search and not args.skip_solve:
        logger.debug("running solve w/cls search")
        do_solve_track(args, track_config, cls=True, search=True,
                       validation=True)

    # . solve without classifier search validation
    if not args.skip_search and not args.skip_solve:
        logger.debug("running solve wo/cls search")
        do_solve_track(args, track_config, cls=False, search=True,
                       validation=True)

    # . solve write ssvm validation
    if not args.skip_write_ssvm and not args.skip_ssvm:
        do_solve_write_ssvm(args, track_config, cls=True)

    # . evaluate with classifier search validation
    if not args.skip_search:
        logger.debug("running eval w/cls search")
        do_eval_track(args, track_config, cls=True, search=True,
                      validation=True)

    # # . evaluate without classifier search validation
    if not args.skip_search:
        logger.debug("running eval wo/cls search")
        do_eval_track(args, track_config, cls=False, search=True,
                      validation=True)

    # . solve compute ssvm validation
    if args.compute_ssvm:
        do_solve_compute_ssvm(args, track_config, cls=True, local=args.local)

    # . predict tracking test
    if not args.skip_predict_tracking:
        do_predict_track(args, track_config, validation=False)

    # . extract edges tracking test
    if not args.skip_extract:
        do_extract_track(args, track_config, validation=False)

    # . predict classifier database test
    if not args.skip_predict_cls:
        os.chdir(os.path.join(paths_machine['HOME'], "classifier_setups",
                              "celegans_setups"))
        os.environ['LD_LIBRARY_PATH'] = environ_lib_path_tf
        os.environ['PATH'] = environ_path_tf
        do_predict_cls(args, cls_config, validation=False)

        os.chdir(os.path.join(paths_machine['HOME'], "unet_setups",
                              "celegans_setups"))

        os.environ['LD_LIBRARY_PATH'] = environ_lib_path_torch
        os.environ['PATH'] = environ_path_torch

    # . solve greedy test
    if not args.skip_greedy and not args.skip_solve:
        do_solve_greedy(args, track_config)

    # . evaluate greedy test
    for fr in frames:
        results.setdefault("greedy", {})[fr] = \
            do_eval_greedy(args, track_config, frames=fr)
    logger.info("greedy results: %s", filter_results(results["greedy"]))

    # . solve with classifier test
    if not args.skip_solve_test and not args.skip_solve:
        logger.debug("running solve w/cls")
        do_solve_track(args, track_config, cls=True, search=False,
                       validation=False)

    # . evaluate with classifier test
    logger.debug("running eval w/cls")
    for fr in frames:
        results.setdefault("w/cls", {})[fr] = \
            do_eval_track(args, track_config, cls=True,
                          search=False, validation=False,
                          frames=fr)
    logger.info("w/cls results: %s", filter_results(results["w/cls"]))

    # . solve without classifier test
    if not args.skip_solve_test and not args.skip_solve:
        logger.debug("running solve wo/cls")
        do_solve_track(args, track_config, cls=False, search=False,
                       validation=False)

    # . evaluate without classifier test
    logger.debug("running eval wo/cls")
    for fr in frames:
        results.setdefault("wo/cls", {})[fr] = \
            do_eval_track(args, track_config, cls=False,
                          search=False, validation=False,
                          frames=fr)
    logger.info("wo/cls results: %s", filter_results(results["wo/cls"]))

    get_and_set_best_cls_params(cls_config, track_config, validation=True,
                                all_th=False)
    # . solve ssvm test
    success_ssvm = True
    if not args.skip_solve and not args.skip_ssvm:
        success_ssvm = do_solve_ssvm(args, track_config, cls=True)
        if not success_ssvm:
            logger.warning("unable to get ssvm solve args, "
                           "skipping remaining ssvm steps")

    # . evaluate ssvm test
    if success_ssvm and not args.skip_ssvm:
        for fr in frames:
            results.setdefault("ssvm", {})[fr] = \
                do_eval_ssvm(args, track_config, cls=True, frames=fr)
        logger.info("ssvm results: %s", filter_results(results["ssvm"]))

    if not results["ssvm"]:
        logger.warning("unable to solve/eval ssvm")
    logger.debug("done")

    logger.debug("results first sample:")
    print("|              |  fp |  fn |  id | fp_div | fn_div | sum_div |"
          " sum | DET | TRA |   REFT |     NR |     ER |    GT |")
    for fr in frames:
        for k, resT in results.items():
            res = resT[fr][0]
            sum_errors = (res['fp_edges'] + res['fn_edges'] +
                          res['identity_switches'] +
                          res['fp_divisions'] + res['fn_divisions'])
            sum_divs = res['fp_divisions'] + res['fn_divisions']
            reft = res["num_error_free_tracks"]/res["num_gt_cells_last_frame"]
            print("| {:>6} {:3d}fr | {:3d} | {:3d} | {:3d} |"
                  "    {:3d} |    {:3d} |     {:3d} | {:3d} |     |     |"
                  " {:.4f} | {:.4f} | {:.4f} | {:5d} |".format(
                k, fr,
                int(res['fp_edges']), int(res['fn_edges']),
                int(res['identity_switches']),
                int(res['fp_divisions']), int(res['fn_divisions']),
                int(sum_divs),
                int(sum_errors),
                reft, res['node_recall'], res['edge_recall'],
                int(res['gt_edges'])))
