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
    parser.add_argument('-c', '--config', required=True,
                        help='path to base config')
    parser.add_argument('--no_augment', dest="augment", action="store_false",
                        help='no augmentation')
    parser.add_argument('--tf_data', action="store_true",
                        help='use tf_data')
    parser.add_argument('--amp', action="store_true",
                        help='use mixed precision')
    parser.add_argument('--input_shape', type=str, default=None,
                        help='input shape to network')
    parser.add_argument('--num_fmaps', type=int, default=None,
                        help='number of feature maps after first conv, for vgg')
    parser.add_argument('--fmap_inc_factors', type=str, default=None,
                        help='inc. factors of feature maps after blocks, for vgg')
    parser.add_argument('--fc_size', type=int, default=None,
                        help='size of final fc layer, for vgg')
    parser.add_argument('--use_resnet', action="store_true",
                        help='use resnet or vgg')
    parser.add_argument('--resnet_size', type=int, default=None,
                        help='size of resnet, for resnet')
    parser.add_argument('--num_blocks', type=str, default=None,
                        help='number of blocks per stage, for resnet')
    parser.add_argument('--use_bottleneck', action="store_true",
                        help='use bottleneck blocks, for resnet')
    parser.add_argument('--padding', type=str, default=None,
                        choices=['same', 'valid'],
                        help='type of padding, same or valid')
    parser.add_argument('--no_batch_norm', dest="batch_norm", action="store_false",
                        help='use batch norm layers')
    parser.add_argument('--no_make_iso', dest="make_iso", action="store_false",
                        help='make volume isotropic (dont downscale z in the beginning)')
    parser.add_argument('--no_conv4d', dest="conv4d", action="store_false",
                        help='use conv4d layers')
    parser.add_argument('--iterations', type=int, default=None,
                        help='number of iterations')
    parser.add_argument('--checkpoints', type=int, default=None,
                        help='when to store checkpoints')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='mini batch size')

    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=None,
                        help='momentum')
    parser.add_argument('--opt', type=str, default=None,
                        help='tf name optimizer')

    parser.add_argument('--val_checkpoints', type=str, default=None,
                        help='validation checkpoints')

    parser.add_argument('-r', '--root', dest='root_folder', default="experiments",
                        help='Experiment folder to store results.')
    parser.add_argument('-d', '--do', dest='tasks', default=['mknet', 'train', 'validate_checkpoints'], nargs='+',
                        choices=['all',
                                 'mknet',
                                 'train',
                                 'validate_checkpoints',
                                 'evaluate',
                                 'predict',
                                 'match_gt',
                                 ],
                        help='Task to do for experiment.')
    parser.add_argument('--slowpoke', action="store_true",
                        help='use slowpoke instead of tesla queue')
    parser.add_argument('--no_interactive', dest="interactive", action="store_false",
                        help='start interactive/blocking job')

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()

    config = toml.load(args.config)
    if not args.augment:
        del config['training']['augmentation']
    if args.tf_data:
        config['training']['use_tf_data'] = True
    if args.amp:
        config['training']['auto_mixed_precision'] = True

    if args.input_shape is not None:
        config['model']['input_shape'] = json.loads(args.input_shape)
    if args.use_resnet:
        if args.resnet_size is not None:
            config['model']['resnet_size'] = args.resnet_size
        else:
            if args.num_blocks is not None:
                config['model']['num_blocks'] = json.loads(args.num_blocks)
            config['model']['use_bottleneck'] = args.use_bottleneck
    else:
        if args.num_fmaps is not None:
            config['model']['num_fmaps'] = args.num_fmaps
        if args.fmap_inc_factors is not None:
            config['model']['fmap_inc_factors'] = json.loads(args.fmap_inc_factors)
        if args.fc_size is not None:
            config['model']['fc_size'] = args.fc_size
    if args.padding is not None:
        config['model']['padding'] = args.padding
    config['model']['use_batchnorm'] = args.batch_norm
    config['model']['make_iso'] = args.make_iso
    config['model']['use_conv4d'] = args.conv4d
    if args.iterations is not None:
        config['model']['max_iterations'] = args.iterations
    if args.checkpoints is not None:
        config['model']['checkpoints'] = args.checkpoints
    if args.batch_size is not None:
        config['model']['batch_size'] = args.batch_size

    if args.lr is not None:
        config['optimizer']['args']['learning_rate'] = args.lr
    if args.opt is not None:
        config['optimizer']['optimizer'] = args.opt
        if args.opt != "AdamOptimizer":
            config['optimizer']['kwargs'] = {}
        if args.momentum is not None:
            config['optimizer']['kwargs']['momentum'] = args.momentum

    if args.val_checkpoints is not None:
        config['validation']['checkpoints'] = json.loads(args.val_checkpoints)

    new_conf_name = os.path.join("tmp_configs",
                                 "config" + datetime.now().strftime('%y%m%d_%H%M%S') + ".toml")
    with open(new_conf_name, 'w') as f:
        toml.dump(config, f)

    cmd = ["bsub"]
    if args.interactive:
        cmd.append("-I")
    else:
        cmd.append("-e")
        cmd.append("-logs/%J.error")
        cmd.append("-o")
        cmd.append("logs/%J.out")
    cmd.append("-n")
    cmd.append("5")
    cmd.append("-q")
    if args.slowpoke:
        cmd.append("slowpoke")
    else:
        cmd.append("gpu_rtx")
    cmd.append("-gpu")
    cmd.append("num=1:mps=no")
    cmd.append('-R"rusage[mem=25600]"')
    cmd.append("python")
    cmd.append("run_lcdc.py")
    cmd.append("-c")
    cmd.append(new_conf_name)
    cmd.append("-r")
    cmd.append(args.root_folder)
    expid = "classifier_" + datetime.now().strftime('%y%m%d_%H%M%S')
    cmd.append("-id")
    cmd.append(expid)
    cmd.append("-d")
    cmd.append(" ".join(args.tasks))
    print(cmd)
    subprocess.run(" ".join(cmd), check=True, shell=True)

if __name__ == "__main__":
    main()
