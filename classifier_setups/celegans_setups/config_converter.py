from copy import deepcopy
import os
import sys

import toml

if __name__ == "__main__":

    old_config = toml.load(sys.argv[1])
    new_config = {}

    new_config['general'] = deepcopy(old_config['general'])
    if 'overwrite' in new_config['general']:
        del new_config['general']['overwrite']
    if 'queue' in new_config['general']:
        del new_config['general']['queue']
    if 'debug' in new_config['general']:
        del new_config['general']['debug']

    new_config['general']['setup_dir'] = os.path.dirname(sys.argv[1])

    new_config['model'] = deepcopy(old_config['model'])
    new_config['model']['path_to_script'] = "/groups/funke/home/hirschp/linajea_experiments/classifier_setups/setup_class_3XX/mknet.py"

    if old_config['model']['use_resnet']:
        new_config['model']['network_type'] = 'resnet'
        del new_config['model']['use_resnet']
        if 'use_efficientnet' in new_config['model']:
            del new_config['model']['use_efficientnet']
        if 'efficientnet_size' in new_config['model']:
            del new_config['model']['efficientnet_size']
        if 'fmap_inc_factors' in new_config['model']:
            del new_config['model']['fmap_inc_factors']
        if 'downsample_factors' in new_config['model']:
            del new_config['model']['downsample_factors']
        if 'kernel_sizes' in new_config['model']:
            del new_config['model']['kernel_sizes']
        if 'fc_size' in new_config['model']:
            del new_config['model']['fc_size']
    elif old_config['model']['use_efficientnet']:
        new_config['model']['network_type'] = 'efficientnet'
        del new_config['model']['use_efficientnet']
        if 'fmap_inc_factors' in new_config['model']:
            del new_config['model']['fmap_inc_factors']
        if 'downsample_factors' in new_config['model']:
            del new_config['model']['downsample_factors']
        if 'kernel_sizes' in new_config['model']:
            del new_config['model']['kernel_sizes']
        if 'fc_size' in new_config['model']:
            del new_config['model']['fc_size']
        if 'num_fmaps' in new_config['model']:
            del new_config['model']['num_fmaps']
        if 'use_resnet' in new_config['model']:
            del new_config['model']['use_resnet']
        if 'num_blocks' in new_config['model']:
            del new_config['model']['num_blocks']
        if 'use_bottleneck' in new_config['model']:
            del new_config['model']['use_bottleneck']
        if 'resnet_size' in new_config['model']:
            del new_config['model']['resnet_size']
    else:
        new_config['model']['network_type'] = 'vgg'
        if 'use_efficientnet' in new_config['model']:
            del new_config['model']['use_efficientnet']
        if 'efficientnet_size' in new_config['model']:
            del new_config['model']['efficientnet_size']
        if 'use_resnet' in new_config['model']:
            del new_config['model']['use_resnet']
        if 'num_blocks' in new_config['model']:
            del new_config['model']['num_blocks']
        if 'use_bottleneck' in new_config['model']:
            del new_config['model']['use_bottleneck']
        if 'resnet_size' in new_config['model']:
            del new_config['model']['resnet_size']
    if 'dropout' in new_config['model']:
        del new_config['model']['dropout']
        new_config['model']['use_dropout'] = old_config['model']['dropout']
    if 'global_pool' in new_config['model']:
        del new_config['model']['global_pool']
        new_config['model']['use_global_pool'] = old_config['model']['global_pool']
    if 'make_iso' in new_config['model']:
        del new_config['model']['make_iso']
        new_config['model']['make_isotropic'] = old_config['model']['make_iso']

    if 'batch_size' in new_config['model']:
        del new_config['model']['batch_size']
    if 'num_gpus' in new_config['model']:
        del new_config['model']['num_gpus']
    if 'num_workers' in new_config['model']:
        del new_config['model']['num_workers']
    if 'cache_size' in new_config['model']:
        del new_config['model']['cache_size']
    if 'max_iterations' in new_config['model']:
        del new_config['model']['max_iterations']
    if 'checkpoints' in new_config['model']:
        del new_config['model']['checkpoints']
    if 'snapshots' in new_config['model']:
        del new_config['model']['snapshots']
    if 'profiling' in new_config['model']:
        del new_config['model']['profiling']


    if 'classes' in old_config['data']:
        new_config['model']['classes'] = old_config['data']['classes']
    if 'class_ids' in old_config['data']:
        new_config['model']['class_ids'] = old_config['data']['class_ids']


    new_config['train'] = deepcopy(old_config['training'])
    new_config['train']['path_to_script'] = "/groups/funke/home/hirschp/linajea_experiments/classifier_setups/setup_class_3XX/train.py"

    if 'linajea_config' in new_config['train']:
        del new_config['train']['linajea_config']
    if 'roi' in new_config['train']:
        del new_config['train']['roi']
    if 'use_database' in new_config['train']:
        del new_config['train']['use_database']
    if 'auto_mixed_precision' in new_config['train']:
        del new_config['train']['auto_mixed_precision']
        new_config['train']['use_auto_mixed_precision'] = old_config['training']['auto_mixed_precision']
    new_config['train']['batch_size'] = old_config['model']['batch_size']
    new_config['train']['cache_size'] = old_config['model']['cache_size']
    new_config['train']['max_iterations'] = old_config['model']['max_iterations']
    new_config['train']['checkpoint_stride'] = old_config['model']['checkpoints']
    new_config['train']['snapshot_stride'] = old_config['model']['snapshots']
    new_config['train']['profiling_stride'] = old_config['model']['profiling']

    new_config['train']['job'] = {
        'num_workers': 1,
        'queue': 'gpu_tesla'
    }

    if 'augmentation' in new_config['train']:
        new_config['train']['augment'] = new_config['train']['augmentation']
        del new_config['train']['augmentation']
        for noise in ['noise', 'noise2']:
            if noise in new_config['train']['augment']:
                if new_config['train']['augment'][noise]['mode'] == 'gaussian':
                    new_config['train']['augment']['noise_gaussian'] = {
                        'var':
                        new_config['train']['augment'][noise]['var']}

                elif new_config['train']['augment'][noise]['mode'] == 's&p':
                    new_config['train']['augment']['noise_saltpepper'] = {
                        'amount':
                        new_config['train']['augment'][noise]['amount']}
                else:
                    raise RuntimeError("unknown noise mode! check manually")
                del new_config['train']['augment'][noise]

        if 'min_key' in old_config['data']:
            new_config['train']['augment']['min_key'] = old_config['data']['min_key']
        if 'max_key' in old_config['data']:
            new_config['train']['augment']['max_key'] = old_config['data']['max_key']
        if 'norm_min' in old_config['data']:
            new_config['train']['augment']['norm_min'] = old_config['data']['norm_min']
        if 'norm_max' in old_config['data']:
            new_config['train']['augment']['norm_max'] = old_config['data']['norm_max']

    new_config['optimizerTF1'] = deepcopy(old_config['optimizer'])


    new_config['evaluate'] = deepcopy(old_config['evaluation'])
    del new_config['evaluate']['distance_limit']
    new_config['evaluate']['parameters'] = {
        'matching_threshold': {
            '50': 22,
            '1000': 11
        },
        'roi': deepcopy(old_config['evaluation']['roi'])
    }
    if 'masked' in new_config['evaluate']:
        del new_config['evaluate']['masked']
    if 'use_database' in new_config['evaluate']:
        del new_config['evaluate']['use_database']
    if 'roi' in new_config['evaluate']:
        del new_config['evaluate']['roi']


    new_config['predict'] = deepcopy(old_config['prediction'])
    if 'use_database' in new_config['predict']:
        del new_config['predict']['use_database']
    new_config['predict']['path_to_script'] = "/groups/funke/home/hirschp/linajea_experiments/classifier_setups/setup_class_3XX/predict.py"
    new_config['predict']['job'] = {
        'num_workers': 4,
        'queue': 'gpu_rtx'
    }


    new_config['train_data'] = {}
    new_config['train_data']['use_database'] = old_config['training']['use_database']
    new_config['train_data']['roi'] = deepcopy(old_config['training']['roi'])
    if new_config['train_data']['use_database']:
        new_config['train_data']['db_meta_info'] = {
            'setup_dir': old_config['postprocessing']['linajea_config']['general']['setup_dir'],
            'checkpoint': old_config['postprocessing']['linajea_config']['prediction']['iteration'],
            'cell_score_threshold': old_config['postprocessing']['linajea_config']['prediction']['cell_score_threshold']
        }
    new_config['train_data']['data_sources'] = []
    for d in old_config['data']['train_data_dirs']:
        new_config['train_data']['data_sources'].append({
            'datafile': {
                'filename': d,
                'group': old_config['data']['raw_key']
        }})
    if old_config['data']['voxel_size']:
        new_config['train_data']['voxel_size'] = old_config['data']['voxel_size']


    new_config['test_data'] = {}
    new_config['test_data']['use_database'] = old_config['prediction']['use_database']
    new_config['test_data']['checkpoint'] = old_config['validation']['checkpoints'][-1]
    new_config['test_data']['prob_threshold'] = old_config['validation']['prob_threshold'][-1]
    new_config['test_data']['skip_predict'] = False
    new_config['test_data']['roi'] = deepcopy(old_config['postprocessing']['roi'])
    if new_config['test_data']['use_database']:
        new_config['test_data']['db_meta_info'] = {
            'setup_dir': old_config['postprocessing']['linajea_config']['general']['setup_dir'],
            'checkpoint': old_config['postprocessing']['linajea_config']['prediction']['iteration'],
            'cell_score_threshold': old_config['postprocessing']['linajea_config']['prediction']['cell_score_threshold']
        }
    new_config['test_data']['data_sources'] = []
    for d in old_config['data']['test_data_dirs']:
        new_config['test_data']['data_sources'].append({
            'datafile': {
                'filename': d,
                'group': old_config['data']['raw_key']
        }})
    if old_config['data']['voxel_size']:
        new_config['test_data']['voxel_size'] = old_config['data']['voxel_size']


    new_config['validate_data'] = {}
    new_config['validate_data']['use_database'] = old_config['prediction']['use_database']
    new_config['validate_data']['prob_thresholds'] = deepcopy(old_config['validation']['prob_threshold'])
    new_config['validate_data']['checkpoints'] = deepcopy(old_config['validation']['checkpoints'])
    new_config['validate_data']['skip_predict'] = False
    new_config['validate_data']['roi'] = deepcopy(old_config['postprocessing']['roi'])
    if new_config['validate_data']['use_database']:
        new_config['validate_data']['db_meta_info'] = {
            'setup_dir': old_config['postprocessing']['linajea_config']['general']['setup_dir'],
            'checkpoint': old_config['postprocessing']['linajea_config']['prediction']['iteration'],
            'cell_score_threshold': old_config['postprocessing']['linajea_config']['prediction']['cell_score_threshold']
        }
    new_config['validate_data']['data_sources'] = []
    for d in old_config['data']['val_data_dirs']:
        new_config['validate_data']['data_sources'].append({
            'datafile': {
                'filename': d,
                'group': old_config['data']['raw_key']
        }})
    if old_config['data']['voxel_size']:
        new_config['validate_data']['voxel_size'] = old_config['data']['voxel_size']


    print(new_config)

    out_fn = os.path.splitext(sys.argv[1])[0] + "2.toml"
    print(out_fn)
    with open(out_fn, 'w') as f:
        toml.dump(new_config, f)
