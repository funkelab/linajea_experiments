import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time

import attr
import toml

# import daisy
from funlib.run import run
from linajea.config import (TrackingConfig,
                            maybe_fix_config_paths_to_machine_and_load)

logger = logging.getLogger(__name__)


def backup_and_copy_file(source, target, fn):
    target_fn = os.path.join(target, fn)
    if os.path.exists(target_fn):
        os.makedirs(os.path.join(target, "backup"), exist_ok=True)
        shutil.copy2(target_fn,
                     os.path.join(target, "backup", fn + "_backup" + str(int(time.time()))))
    if source is not None:
        source_fn = os.path.join(source, fn)
        if source_fn != target_fn:
            shutil.copy2(source_fn, target_fn)


def do_mknet(args, config, cmd):
    queue = 'interactive' if args.interactive else 'local'
    num_gpus = 0
    num_cpus = 1
    run_cmd(args, config, cmd, "mknet",
            queue, num_gpus, num_cpus)

def do_train(args, config, cmd):
    queue = config.train.job.queue
    num_gpus = 1
    num_cpus = config.train.job.num_workers
    flags = (['-P', config.train.job.lab]
             if config.train.job.lab is not None else [])
    run_cmd(args, config, cmd, "train",
            queue, num_gpus, num_cpus,
            flags=flags)

def do_predict(args, config, cmd):
    queue = 'interactive' if args.interactive else 'local'
    num_gpus = 0
    num_cpus = config.predict.job.num_workers
    flags = (['-P', config.predict.job.lab]
             if config.predict.job.lab is not None else [])
    run_cmd(args, config, cmd, "predict",
            queue, num_gpus, num_cpus,
            flags=flags)

def do_extract_edges(args, config, cmd):
    queue = 'interactive' if args.interactive else config.extract.job.queue
    num_gpus = 0
    num_cpus = config.extract.job.num_workers
    flags = (['-P', config.extract.job.lab]
             if config.extract.job.lab is not None else [])
    run_cmd(args, config, cmd, "extract",
            queue, num_gpus, num_cpus,
            flags=flags)

def do_solve(args, config, cmd, wait=True):
    queue = 'interactive' if args.interactive else config.solve.job.queue
    num_gpus = 0
    num_cpus = config.solve.job.num_workers
    flags = (['-P ' + config.solve.job.lab]
             if config.solve.job.lab is not None else [])
    flags.append('-W 60')

    if args.val_param_id is not None:
        cmd += ["--val_param_id", str(args.val_param_id)]
    if args.array_job:
        array_limit = 100
        if args.param_ids is not None:
            array_start = args.param_ids[0]
            array_end = args.param_ids[1]
            cmd += ["--param_id", '"$LSB_JOBINDEX"']
        else:
            array_start = 1
            array_end = len(config.solve.parameters)
            cmd += ["--param_list_idx", '"$LSB_JOBINDEX"']
    else:
        array_limit = None
        array_start = None
        array_end = None

        if args.param_ids is not None:
            cmd += ["--param_ids",
                    str(args.param_ids[0]),
                    str(args.param_ids[1])]
        elif args.param_id is not None:
            cmd += ["--param_id", str(args.param_id)]

    # if args.run_evaluate and (args.array_job or args.eval_array_job):
    if args.run_evaluate:
        wait = False

    jobid = run_cmd(args, config, cmd, "solve",
            queue, num_gpus, num_cpus,
            array_limit=array_limit,
            array_start=array_start, array_end=array_end,
            flags=flags, wait=wait)
    return jobid

def do_evaluate(args, config, cmd, jobid=None, wait=True):
    queue = 'interactive' if args.interactive else config.evaluate.job.queue
    num_gpus = 0
    num_cpus = 1
    flags = (['-P', config.evaluate.job.lab]
             if config.evaluate.job.lab is not None else [])
    flags.append('-W 30')

    if args.param_id is not None:
        cmd += ["--param_id", str(args.param_id)]
    if args.val_param_id is not None:
        cmd += ["--val_param_id", str(args.val_param_id)]

    if jobid is not None:
        if args.array_job:
            flags.extend(['-w', 'done({}[*])'.format(jobid)])
        else:
            flags.extend(['-w', 'done({})'.format(jobid)])

    if args.array_job or args.eval_array_job:
        array_limit = 100
        if args.param_ids is not None:
            array_start = args.param_ids[0]
            array_end = args.param_ids[1]
            cmd += ["--param_id", '"$LSB_JOBINDEX"']
        else:
            array_start = 1
            array_end = len(config.solve.parameters)
            cmd += ["--param_list_idx", '"$LSB_JOBINDEX"']
    else:
        array_limit = None,
        array_start = None
        array_end = None
    run_cmd(args, config, cmd, "evaluate",
            queue, num_gpus, num_cpus,
            array_limit=array_limit,
            array_start=array_start, array_end=array_end,
            flags=flags, wait=wait)


def run_cmd(args, config, cmd, job_name,
            queue, num_gpus, num_cpus,
            array_limit=0, array_size=0,
            array_start=None, array_end=None, flags=[], wait=True):
    if not args.local and not args.slurm and not args.gridengine:
        cmd = run(
            command=cmd,
            queue=queue,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            singularity_image=config.general.singularity_image,
            batch=not args.interactive,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            job_name=job_name,
            array_size=array_size,
            array_limit=array_limit,
            array_start=array_start,
            array_end=array_end,
            flags=flags)
    print(cmd)
    print(' '.join(cmd))


    if not args.array_job and not args.eval_array_job and (args.slurm or args.gridengine):
        if args.slurm:
            if num_gpus > 0:
                cmd = ['sbatch', '../run_slurm_gpu.sh'] + cmd[1:]
            else:
                cmd = ['sbatch', '../run_slurm_cpu.sh'] + cmd[1:]
        elif args.gridengine:
            if num_gpus > 0:
                cmd = ['qsub', '../run_gridengine_gpu.sh'] + cmd[1:]
            else:
                cmd = ['qsub', '../run_gridengine_cpu.sh'] + cmd[1:]
        print(cmd)
        output = subprocess.run(cmd, check=True)
    else:
        output = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='UTF-8')

        jobid = None
        if not args.local:
            bsub_stdout_regex = re.compile("Job <(\d+)> is submitted*")
            logger.debug("Command output: %s" % output)
            print(output.stdout)
            print(output.stderr)
            match = bsub_stdout_regex.match(output.stdout)
            jobid = match.group(1)
            print(jobid)
            if wait and \
               not subprocess.run(["bwait", "-w", 'ended({})'.format(jobid)]):
                print("{} failed".format(cmd))
                exit()

        return jobid


if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to (post)process')
    parser.add_argument('--mknet', action="store_true",
                        dest='run_mknet', help='run mknet?')
    parser.add_argument('--train', action="store_true",
                        dest='run_train', help='run train?')
    parser.add_argument('--predict', action="store_true",
                        dest='run_predict', help='run predict?')
    parser.add_argument('--extract_edges', action="store_true",
                        dest='run_extract_edges',
                        help='run extract edges?')
    parser.add_argument('--solve', action="store_true",
                        dest='run_solve', help='run solve?')
    parser.add_argument('--evaluate', action="store_true",
                        dest='run_evaluate', help='run evaluate?')
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    parser.add_argument('--param_id', type=int, default=None,
                        help='eval parameters_id')
    parser.add_argument('--val_param_id', type=int, default=None,
                        help='use validation parameters_id')
    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')
    parser.add_argument("--local", action="store_true",
                        help='run locally or on cluster?')
    parser.add_argument("--slurm", action="store_true",
                        help='run on slurm cluster?')
    parser.add_argument("--gridengine", action="store_true",
                        help='run on gridengine cluster?')
    parser.add_argument("--interactive", action="store_true",
                        help='run on interactive node on cluster?')
    parser.add_argument('--array_job', action="store_true",
                        help='submit each parameter set for solving/eval as one job?')
    parser.add_argument('--eval_array_job', action="store_true",
                        help='submit each parameter set for eval as one job?')
    parser.add_argument('--param_ids', default=None, nargs=2,
                        help='start/end range of eval parameters_ids')
    parser.add_argument('--wait_job_id', type=str, default=None,
                        help='wait for this job before starting')
    parser.add_argument("--no_block_after_eval", dest="block_after_eval",
                        action="store_false",
                        help='block after starting eval jobs?')


    args = parser.parse_args()
    config = maybe_fix_config_paths_to_machine_and_load(args.config)
    config = TrackingConfig(**config)
    setup_dir = config.general.setup_dir
    lin_exp_root = config.model.path_to_script.split("/unet_setups")[0]

    source_dir = os.path.dirname(os.path.abspath(__file__))
    is_new_run = not os.path.exists(setup_dir)
    os.makedirs(setup_dir, exist_ok=True)
    os.makedirs(os.path.join(setup_dir, "tmp_configs"), exist_ok=True)
    backup_and_copy_file(source_dir, setup_dir, 'mknet_celegans.py')
    backup_and_copy_file(source_dir, setup_dir, 'train_val_celegans.py')
    backup_and_copy_file(source_dir, setup_dir, 'train_val_celegans_torch.py')
    backup_and_copy_file(source_dir, setup_dir, 'predict_celegans.py')
    backup_and_copy_file(source_dir, setup_dir, 'write_cells_celegans.py')

    if not is_new_run and \
       os.path.dirname(os.path.abspath(args.config)) != \
           os.path.abspath(setup_dir) and \
           "tmp_configs" not in args.config:
        raise RuntimeError(
            "overwriting config with external config file (%s - %s)",
            args.config, setup_dir)
    config_file = os.path.basename(args.config)
    backup_and_copy_file(os.path.dirname(args.config),
                         setup_dir,
                         os.path.basename(args.config))
    if is_new_run:
        config.path = os.path.join(setup_dir, os.path.basename(args.config))

    os.chdir(setup_dir)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    if args.run_mknet:
        cmd = ["python", config.model.path_to_script,
               "--config", config.path]
        do_mknet(args, config, cmd)

    if args.run_train:
        cmd = ["python", config.train.path_to_script,
               "--config", config.path]
        do_train(args, config, cmd)


    os.environ["GRB_LICENSE_FILE"] = "/misc/local/gurobi-9.1.2/gurobi.lic"
    run_post_steps = []
    if args.run_predict:
        run_post_steps.append("01_predict_blockwise.py")
    if args.run_extract_edges:
        run_post_steps.append("02_extract_edges.py")
    if args.run_solve:
        run_post_steps.append("03_solve.py")
    if args.run_evaluate:
        run_post_steps.append("04_evaluate.py")


    config.path = os.path.join("tmp_configs", "config_{}.toml".format(
        time.time()))
    config_dict = attr.asdict(config)
    if config.solve.parameters_search is not None:
        del config_dict['solve']['parameters_search']
    with open(config.path, 'w') as f:
        toml.dump(config_dict, f)

    jobid = args.wait_job_id
    for post_step in run_post_steps:
        cmd = ["python",
               os.path.join(lin_exp_root, "run_scripts", post_step)]
        if args.checkpoint > 0:
            cmd.append("--checkpoint")
            cmd.append(str(args.checkpoint))
        if args.validation:
            cmd.append("--validation")
        if args.validate_on_train:
            cmd.append("--validate_on_train")

        cmd += ["--config", config.path]

        if post_step == "01_predict_blockwise.py":
            do_predict(args, config, cmd)
        elif post_step == "02_extract_edges.py":
            do_extract_edges(args, config, cmd)
        elif post_step == "03_solve.py":
            jobid = do_solve(args, config, cmd)
        elif post_step == "04_evaluate.py":
            # do_evaluate(args, config, cmd, jobid=jobid, wait=args.block_after_eval)


            config_dict = attr.asdict(config)
            if config.solve.parameters_search is not None:
                del config_dict['solve']['parameters_search']

            tmp_path = os.path.join("tmp_configs", "config_{}.toml".format(
                time.time()))
            config_dict['evaluate']['parameters']['roi']['shape'][0] = 200
            with open(tmp_path, 'w') as f:
                toml.dump(config_dict, f)
            cmd[cmd.index("--config")+1] = tmp_path
            do_evaluate(args, config, cmd, jobid=jobid, wait=args.block_after_eval)

            tmp_path = os.path.join("tmp_configs", "config_{}.toml".format(
                time.time()))
            config_dict['evaluate']['parameters']['roi']['shape'][0] = 270
            with open(tmp_path, 'w') as f:
                toml.dump(config_dict, f)
            cmd[cmd.index("--config")+1] = tmp_path
            do_evaluate(args, config, cmd, jobid=jobid, wait=args.block_after_eval)

        else:
            raise RuntimeError("invalid postprocessing step! %s", post_step)
