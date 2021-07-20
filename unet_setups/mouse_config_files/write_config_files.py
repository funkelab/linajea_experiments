import json
import argparse

base_config = {
    "data_dir": "../01_data/140521",
    "zarr_file": "140521.zarr",
    "tracks_file": "extended_tracks.txt",
    "daughter_cells_file": "daughter_cells.txt",
    "division_tracks_file": "division_tracks.txt",
    "train_input_shape": [7, 60, 148, 148],
    "predict_input_shape": [7, 80, 260, 260]
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

    if train_on == 'early':
        config['exclude_times'] = [[225, 275], [400, 450]]
    elif train_on == 'middle':
        config['exclude_times'] = [[50, 100], [400, 450]]
    elif train_on == 'late':
        config['exclude_times'] = [[50, 100], [225, 275]]
    else:
        raise ValueError("Expected 'early' 'middle' or 'late'")

    if setup == '09':
        config['constant_upsample'] = False
        config['average_vectors'] = False
    elif setup == '11':
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

    for setup in ['09', '11']:
        for shift in [False, True]:
            for divisions in [False, True]:
                for train_on in ['early', 'middle', 'late']:
                    write_config_file(
                            setup, shift, divisions, train_on, template)
