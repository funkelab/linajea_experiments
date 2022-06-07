import argparse
import os
import shutil
import subprocess
import sys
import time

import toml

def backup_and_copy_file(source, target, fn):
    target_fn = os.path.join(target, fn)
    if os.path.exists(target_fn):
        os.makedirs(os.path.join(target, "backup"), exist_ok=True)
        os.replace(target_fn,
                   os.path.join(target, "backup", fn + "_backup" + str(int(time.time()))))
    if source is not None:
        source = os.path.join(source, fn)
        shutil.copy2(source, target_fn)


if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--iterations', type=int, default=-1,
                        help='number of iterations to train/checkpoint to predict')
    parser.add_argument('--no_mknet', action="store_false",
                        dest='run_mknet', help='run mknet?')
    parser.add_argument('--no_train', action="store_false",
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
    parser.add_argument('--setup_dir', type=str,
                        required=True, help='output')
    parser.add_argument('--blockwise', action="store_true",
                        help='compute block-wise')
    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')

    args = parser.parse_args()

    source_dir = os.path.dirname(os.path.abspath(__file__))
    lin_exp_root = source_dir.split("/unet_setups")[0]
    if not os.path.exists(args.setup_dir) and args.run_from_exp:
        os.makedirs(args.setup_dir, exist_ok=True)
        backup_and_copy_file(source_dir, args.setup_dir,
                             'mknet_celegans.py')
        backup_and_copy_file(source_dir, args.setup_dir,
                             'train_celegans.py')
        backup_and_copy_file(source_dir, args.setup_dir,
                             'predict_celegans.py')
        backup_and_copy_file(source_dir, args.setup_dir,
                             'write_cells_celegans.py')
    else:
        os.makedirs(args.setup_dir, exist_ok=True)

    backup_and_copy_file(os.path.dirname(args.config),
                         args.setup_dir,
                         os.path.basename(args.config))
    if args.run_from_exp:
        source_dir = args.setup_dir

    os.chdir(args.setup_dir)
    setup_dir = os.getcwd()
    os.makedirs("logs", exist_ok=True)

    if args.run_mknet:
        subprocess.run(
            ["python", os.path.join(source_dir, "mknet_celegans.py"),
             "--config", os.path.join(source_dir,
                                      os.path.basename(args.config)),
             "--setup", setup_dir],
            check=True)

    if args.run_train:
        subprocess.run(
            ["python", os.path.join(source_dir, "train_celegans.py"),
             "--config", os.path.join(source_dir,
                                      os.path.basename(args.config)),
             "--iterations", str(args.iterations),
             "--setup", setup_dir],
            check=True)

    os.environ["GRB_LICENSE_FILE"] = "/misc/local/gurobi-8.0.1/gurobi.lic"
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
               "--config", os.path.join(source_dir,
                                        os.path.basename(args.config)),
               "--setup", setup_dir,
               "--source", source_dir]
        if args.iterations > 0:
            cmd.append("--iteration")
            cmd.append(str(args.iterations))
        if args.validation:
            cmd.append("--validation")
        subprocess.run(cmd,
                       check=True)
