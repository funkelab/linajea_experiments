import json
import argparse

base_config = {
    "data_dir": "../01_data/160328",
    "zarr_file": "160328.zarr",
    "exclude_times": [150, 200],
    "train_input_shape": [7, 124, 124, 124],
    "predict_input_shape": [7, 156, 156, 156]
}


def write_config_file(setup, shift, divisions, train_on, template):
    config = template.copy()
    name = 'setup' + setup
    if shift:
        name = name + '_shift'
    if divisions:
        name = name + '_divisions'
    if not (shift or divisions):
        name = name + '_simple'
    name = name + '_train_' + train_on
    config['setup_dir'] = name
    config['shift'] = shift
    config['divisions'] = divisions
    config['tracks_file'] = 'tracks_' + train_on + '.txt'
    config['daughter_cells_file'] = 'daughter_cells_' + train_on + '.txt'

    if setup == '109':
        config['constant_upsample'] = False
        config['average_vectors'] = False
    elif setup == '111':
        config['constant_upsample'] = True
        config['average_vectors'] = False
    else:
        raise ValueError("expected setup 09 or 11")

    with open(name + '.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to template config file')
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            template = json.load(f)
    else:
        template = base_config

    for setup in ['109', '111']:
        for shift in [False, True]:
            for divisions in [False, True]:
                for train_on in ['side_1', 'side_2']:
                    write_config_file(
                            setup, shift, divisions, train_on, template)
