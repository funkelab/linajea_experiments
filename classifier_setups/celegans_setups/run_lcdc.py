import argparse
from copy import deepcopy
from datetime import datetime
from glob import glob
import functools
import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)
import multiprocessing
import os
import queue
import shutil
import sys
import time
import warnings

import attr
from natsort import natsorted
import numpy as np
import toml

from linajea.config import CellCycleConfig, InferenceDataCellCycleConfig
from linajea import checkOrCreateDB
import mknet
import mknet_tf2
import train_val_prov
import train_val_prov_tf2
import predict
# from eval_node_candidates_par_vec import eval_node_candidates
from eval_node_candidates import eval_node_candidates
import selectGPU

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("once", category=FutureWarning)


logger = logging.getLogger(__name__)


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError('Args to merge_dicts should be dicts')

    for k, v in source.items():
        if isinstance(source[k], dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v

    return sink


def backup_and_copy_file(source, target, fn):
    target_fn = os.path.join(target, fn)
    if os.path.exists(target_fn):
        os.makedirs(os.path.join(target, "backup"), exist_ok=True)
        shutil.copy2(target_fn,
                     os.path.join(target, "backup", fn + "_backup" +
                                  str(int(time.time()))))
    if source is not None:
        source_fn = os.path.join(source, fn)
        if os.path.abspath(source_fn) != os.path.abspath(target_fn):
            shutil.copy2(source_fn, target_fn)


def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        logger.info('time %s: %s', func.__name__, str(datetime.now() - t0))
        return ret
    return wrapper


def fork(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


def fork_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=func,
                                        args=args + (q,), kwargs=kwargs)
            p.start()
            results = None
            while p.is_alive():
                try:
                    results = q.get_nowait()
                except queue.Empty:
                    time.sleep(0.5)
            if p.exitcode == 0 and results is None:
                results = q.get()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
            return results
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


def adapt_config_to_sample(config, sample, inference, inference_data):
    sample = deepcopy(sample)
    if inference.use_database and sample.db_name is None:
        db_meta_info = inference.db_meta_info
        if "polar" in sample.datafile.filename and \
           config.model.with_polar:
            filename = os.path.dirname(sample.datafile.filename)
        else:
            filename = sample.datafile.filename
        sample.db_name = checkOrCreateDB(
            config.general.db_host,
            db_meta_info.setup_dir,
            filename,
            db_meta_info.checkpoint,
            db_meta_info.cell_score_threshold,
            create_if_not_found=False,
            tag=config.general.tag)
        assert sample.db_name is not None, \
            "db not found, config {} sample {}".format(
                db_meta_info, sample)
    inference_data['data_source'] = sample
    config.inference = InferenceDataCellCycleConfig(**inference_data) # type: ignore
    return config


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-r', '--root', dest='root', default=None,
                        help='Experiment folder to store results.')
    parser.add_argument('-id', '--exp-id', dest='expid', default=None,
                        help='ID for experiment.')

    # action options
    parser.add_argument('-d', '--do', dest='do', default=[], nargs='+',
                        choices=['all',
                                 'mknet',
                                 'train',
                                 'validate_checkpoints',
                                 'evaluate',
                                 'predict',
                                 'match_gt',
                                 ],
                        help='Task to do for experiment.')

    parser.add_argument('--checkpoint', dest='checkpoint', default=-1,
                        type=int,
                        help='Specify which checkpoint to use.')

    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')
    parser.add_argument("--validate_on_train", action="store_true",
                        help=('validate using training data'
                              '(to check for overfitting)'))
    parser.add_argument("--swap_val_test", action="store_true",
                        help='swap val and test data')

    parser.add_argument("--dry_run", action="store_true",
                        help="dont change db")
    parser.add_argument("--validation", action="store_true",
                        help="use validation data?")
    parser.add_argument('--val_id', default=-1, type=int,
                        help='id of val params to process')
    parser.add_argument("--tf2", action="store_true",
                        help="use tf2 code")

    args = parser.parse_args()

    return args


def create_folders(args, filebase):
    # create experiment folder
    os.makedirs(filebase, exist_ok=True)

    if args.expid is None and args.run_from_exp:
        setup = "."
        backup_and_copy_file(setup, filebase, 'train.py')
        backup_and_copy_file(setup, filebase, 'mknet.py')
        backup_and_copy_file(setup, filebase, 'vgg.py')
        backup_and_copy_file(setup, filebase, 'util.py')
        backup_and_copy_file(setup, filebase, 'predict.py')
        backup_and_copy_file(setup, filebase, 'eval_node_candidates.py')

    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)

    # create val folders
    val_folder = os.path.join(filebase, 'val')
    os.makedirs(val_folder, exist_ok=True)

    # create test folders
    test_folder = os.path.join(filebase, 'test')
    os.makedirs(test_folder, exist_ok=True)

    return train_folder, val_folder, test_folder


@fork
@time_func
def mk_net(args, config, train_folder):
    if args.tf2:
        mknet_tf2.mk_net(config, train_folder)
    else:
        mknet.mk_net(config, train_folder)


@fork
@time_func
def train_until(args, config, train_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    if args.tf2:
        train_val_prov_tf2.train_until(config, train_folder)
    else:
        train_val_prov.train_until(config, train_folder)


# @fork_return
@fork
@time_func
def predict_nodes(config, checkpoint_file):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    predict.run(config, checkpoint_file)

@fork_return
@time_func
def evaluate(config, output_folder, queue):
    res_fn = os.path.join(
        output_folder,
        "results_ckpt_eval_" + \
        os.path.basename(config.inference.data_source.datafile.filename) + \
        ("_one_off" if config.evaluate.one_off else "") + \
        ("_swa" if config.predict.use_swa else "") + \
        ("_ttr{}".format(config.predict.test_time_reps)
         if config.predict.test_time_reps > 1 else "") + \
        ("_roi{}".format(config.inference.data_source.roi.shape[0])
         if config.inference.data_source.roi.shape[0] != 200 else "") + \
        "_" + str(config.inference.checkpoint) + ".toml")
    if not config.inference.use_database and \
       os.path.isfile(res_fn) and \
       not config.evaluate.force_eval:
        logger.info("result already exists (%s)", res_fn)
        with open(res_fn, 'r') as f:
            results = toml.load(f)
    else:
        results = eval_node_candidates(config, set_gt_db=False)

        if not config.inference.use_database:
            with open(res_fn, 'w') as f:
                toml.dump(results, f)

    logger.info("results %s", results)
    queue.put(results)


@fork
@time_func
def match_gt(config):
    _ = eval_node_candidates(
        config,
        set_gt_db=True)


def validate_checkpoints(args, config, train_folder, val_folder):
    # validate all checkpoints and return best one
    metrics = []
    ckpts = []
    params = []
    results = []

    if args.validate_on_train:
        inference = deepcopy(config.train_data)
        output_folder = train_folder
    else:
        inference = deepcopy(config.validate_data)
        output_folder = val_folder
    if args.checkpoint > 0:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = config.validate_data.checkpoints
    prob_thresholds = config.validate_data.prob_thresholds

    for checkpoint in checkpoints:
        logger.info("config checkpoint: %s", checkpoint)
        checkpoint_file = get_checkpoint_file(
            checkpoint, config.model.net_name, train_folder)

        inference_data = {
            'checkpoint': checkpoint,
            'use_database': inference.use_database,
            'db_meta_info': inference.db_meta_info,
            'force_predict': inference.force_predict
        }
        for idx, prob_th in enumerate(prob_thresholds):
            inference_data['prob_threshold'] = prob_th
            output_folder_t = os.path.join(
                output_folder, str(checkpoint),
                "prob_threshold_{}".format(str(prob_th).replace(".", "_")))
            os.makedirs(output_folder_t, exist_ok=True)

            chkpt_metric = 0
            for sample in inference.data_sources:
                config = adapt_config_to_sample(config, sample, inference,
                                                inference_data)

                if "polar" in config.inference.data_source.datafile.filename and \
                   (not config.model.with_polar or inference.use_database):
                    continue
                if not inference.skip_predict and idx == 0:
                    predict_nodes(config, checkpoint_file)

            # if 'match_gt' in args.do:
            #     for sample in inference.data_sources:
            #         config = adapt_config_to_sample(config, sample, inference, inference_data)
            #         logger.info("setting probable gt state %d", checkpoint)
            #         match_gt(config)

            cnt_samples = 0
            for sample in inference.data_sources:
                config = adapt_config_to_sample(config, sample, inference,
                                                inference_data)
                if "polar" in config.inference.data_source.datafile.filename:
                    continue
                logger.info("evaluating checkpoint %s", checkpoint)
                print(inference.db_meta_info)
                chkpt_metrics = evaluate(config, output_folder_t)

                chkpt_metric += chkpt_metrics['mixed'][config.evaluate.metric]
                cnt_samples += 1
            chkpt_metric /= cnt_samples
            metrics.append(chkpt_metric)
            ckpts.append(checkpoint)
            params.append(prob_th)
            results.append({'checkpoint': checkpoint,
                            'metric': str(chkpt_metric),
                            'prob_threshold': prob_th})
            logger.info("%s checkpoint %6d: %.4f (%s)",
                        config.evaluate.metric,
                        checkpoint, chkpt_metric, prob_th)

    logger.info("%s", config.evaluate.metric)
    for ch, metric, p in zip(ckpts, metrics, params):
        logger.info("%s checkpoint %6d: %.4f (%s)",
                    config.evaluate.metric, ch, metric, p)

    best_checkpoint = ckpts[np.argmax(metrics)]
    best_params = params[np.argmax(metrics)]
    logger.info('best checkpoint: %d', best_checkpoint)
    logger.info('best params: %s', best_params)
    config.test_data.checkpoint = best_checkpoint
    config.test_data.prob_threshold = best_params


def test_checkpoint(args, config, train_folder, test_folder):
    inference = deepcopy(config.test_data)
    prob_threshold = config.test_data.prob_threshold
    if args.checkpoint > 0:
        checkpoint = args.checkpoint
    else:
        checkpoint = config.test_data.checkpoint
    checkpoint_file = get_checkpoint_file(
        checkpoint, config.model.net_name, train_folder)
    output_folder = os.path.join(
                test_folder, str(checkpoint),
                "prob_threshold_{}".format(
                    str(config.test_data.prob_threshold).replace(".", "_")))
    os.makedirs(output_folder, exist_ok=True)

    inference_data = {
        'checkpoint': checkpoint,
        'prob_threshold': prob_threshold,
        'use_database': inference.use_database,
        'db_meta_info': inference.db_meta_info
    }
    for sample in inference.data_sources:
        config = adapt_config_to_sample(config, sample, inference,
                                        inference_data)

        if "polar" in config.inference.data_source.datafile.filename and \
           (not config.model.with_polar or inference.use_database):
            continue
        if 'all' in args.do or 'predict' in args.do:
            logger.info("predicting checkpoint %d", checkpoint)
            if not inference.skip_predict:
                predict_nodes(config, checkpoint_file)
            else:
                logger.info("..skipping")

        # if 'all' in args.do or 'match_gt' in args.do:
        #     logger.info("setting probable gt state %d", checkpoint)
        #     match_gt(config)

    for sample in inference.data_sources:
        config = adapt_config_to_sample(config, sample, inference,
                                        inference_data)

        if "polar" in config.inference.data_source.datafile.filename:
            continue

        if 'all' in args.do or 'evaluate' in args.do:
            logger.info("evaluating checkpoint %s",
                        checkpoint if str(checkpoint) is not None else "last")
            results = evaluate(config, output_folder)
            logger.info("results %s", results)


def get_checkpoint_file(iteration, name, train_folder):
    return os.path.join(train_folder, name + '_checkpoint_%d' % iteration)


def get_checkpoint_list(name, train_folder):
    checkpoints = natsorted(glob(
        os.path.join(train_folder, name + '_checkpoint_*.index')))
    return [int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])
            for cp in checkpoints]


def main():
    # parse command line arguments
    args = get_arguments()

    if not args.do:
        raise ValueError("Provide a task to do (-d/--do)")

    config = CellCycleConfig.from_file(os.path.abspath(args.config))
    assert config.general.setup_dir is not None, \
        "please provide general.setup_dir in config!"
    setup_dir = config.general.setup_dir
    # create folder structure for experiment
    train_folder, val_folder, test_folder = create_folders(args, setup_dir)

    if args.swap_val_test:
        config.test_data.data_sources, config.validate_data.data_sources = \
            config.validate_data.data_sources, config.test_data.data_sources

    if args.dry_run:
        config.evaluate.dry_run = True

    # set logging level
    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler(os.path.join(setup_dir, "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout)
        ])
    logger.info("JOBID %s", os.environ.get("LSB_JOBID"))
    logger.info('attention: using config file %s', args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        selectedGPU = selectGPU.selectGPU()
        if selectedGPU is None:
            logger.warning("no free GPU available!")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selectedGPU)
        logger.info("setting CUDA_VISIBLE_DEVICES to device {}".format(
            selectedGPU))
    else:
        logger.info("CUDA_VISIBILE_DEVICES already set, device {}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]))

    if "tmp_config" not in args.config and "config_1" not in args.config:
        logger.info("backing up config %s", args.config)
        backup_and_copy_file(os.path.dirname(args.config),
                             setup_dir,
                             os.path.basename(args.config))
        # with open(os.path.join(setup_dir, os.path.basename(args.config)), 'w') as f:
        #     toml.dump(attr.asdict(config), f)

    logger.info('used config: %s', config)

    # create network
    if 'all' in args.do or 'mknet' in args.do:
        mk_net(args, config, train_folder)

    # train network
    if 'all' in args.do or 'train' in args.do:
        train_until(args, config, train_folder)

    if 'all' in args.do or 'validate_checkpoints' in args.do:
        validate_checkpoints(args, config, train_folder, val_folder)

    # predict test set
    if any(i in args.do for i in ['all', 'predict', 'evaluate',
                                  'match_gt']):
        assert config.test_data.checkpoint is not None, \
            "checkpoint has to be set by now, by flag or using validate_checkpoints"
        test_checkpoint(args, config, train_folder, test_folder)


if __name__ == "__main__":
    main()
