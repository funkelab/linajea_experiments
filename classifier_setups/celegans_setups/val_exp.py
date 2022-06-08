import argparse
from datetime import datetime
import os
import subprocess
import sys
import time

import json
import toml

def get_arguments():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', required=True,
                        help='path to experiment')

    parser.add_argument('--val_checkpoints', type=str, default=None,
                        help='validation checkpoints')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='run validation on training set')

    parser.add_argument('--rtx', action="store_true",
                        help='use rtx instead of tesla queue')
    parser.add_argument('--no_interactive', dest="interactive", action="store_false",
                        help='start interactive/blocking job')

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()

    cmd = ["bsub"]
    if args.interactive:
        cmd.append("-I")

    cmd.append("-n")
    cmd.append("5")
    cmd.append("-q")
    if args.rtx:
        cmd.append("gpu_rtx")
    else:
        cmd.append("gpu_tesla")
    cmd.append("-gpu")
    cmd.append("num=1:mps=no")
    cmd.append('-R"rusage[mem=25600]"')
    cmd.append("python")
    cmd.append("run_lcdc.py")
    cmd.append("-id")
    cmd.append(args.exp)
    cmd.append("-d")
    cmd.append("validate_checkpoints")
    if args.validate_on_train:
        cmd.append("--validate_on_train")

    conf_name = os.path.join(args.exp, "config.toml")
    if args.val_checkpoints is not None:
        ckpt = json.loads(args.val_checkpoints)
        if isinstance(ckpt, list):
            config = toml.load(conf_name)
            config['validation']['checkpoints'] = ckpt

            conf_name = os.path.join("tmp_configs",
                                     "config" + datetime.now().strftime('%y%m%d_%H%M%S') + ".toml")
            with open(conf_name, 'w') as f:
                toml.dump(config, f)
        else:
            cmd.append("--checkpoint")
            cmd.append(str(ckpt))


    cmd.append("-c")
    cmd.append(conf_name)
    print(cmd)
    subprocess.run(" ".join(cmd), check=True, shell=True)
    time.sleep(2)

if __name__ == "__main__":
    main()
