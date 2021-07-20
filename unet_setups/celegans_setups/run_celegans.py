import argparse
import os
import shutil
import subprocess
import sys
import time

import toml

from linajea.config import TrackingConfig

import logging
logger = logging.getLogger(__name__)

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
    parser.add_argument('--param_id', type=int,
                        help='eval parameters_id')
    parser.add_argument('--val_param_id', type=int,
                        help='use validation parameters_id')

    args = parser.parse_args()
    config = TrackingConfig.from_file(os.path.abspath(args.config))
    setup_dir = config.general.setup_dir
    lin_exp_root = config.model.path_to_script.split("/unet_setups")[0]

    os.makedirs(setup_dir, exist_ok=True)
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
        subprocess.run(
            ["python", config.model.path_to_script,
             "--config", config.path],
            check=True)

    if args.run_train:
        subprocess.run(
            [
                "python", config.train.path_to_script,
                "--config", config.path,
            ],
            check=True)

    os.environ["GRB_LICENSE_FILE"] = "/misc/local/gurobi-9.0.3/gurobi.lic"
    run_post_steps = []
    if args.run_predict:
        run_post_steps.append("01_predict_blockwise.py")
    if args.run_extract_edges:
        run_post_steps.append("02_extract_edges.py")
    if args.run_solve:
        run_post_steps.append("03_solve.py")
    if args.run_evaluate:
        run_post_steps.append("04_evaluate.py")

    for post_step in run_post_steps:
        cmd = ["python",
               os.path.join(lin_exp_root, "run_scripts", post_step),
               "--config", config.path]
        if args.checkpoint > 0:
            cmd.append("--checkpoint")
            cmd.append(str(args.checkpoint))
        if args.validation:
            cmd.append("--validation")
        if post_step == "03_solve.py":
            if args.val_param_id is not None:
                cmd += ["--val_param_id", str(args.val_param_id)]
        if post_step == "04_evaluate.py":
            if args.param_id is not None:
                cmd += ["--param_id", str(args.param_id)]
        subprocess.run(cmd,
                       check=True)
