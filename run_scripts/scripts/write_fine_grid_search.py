import json
import sys
import os


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python write_config_files.py <sample_config> "
              "<best_config_and_neighborhoods_json> <out_dir>")
        exit(1)
    config_file = sys.argv[1]
    grid_search_vals = sys.argv[2]
    out_dir = sys.argv[3]
    with open(config_file, 'r') as f:
        config = json.load(f)

    with open(grid_search_vals, 'r') as f:
        grid_search = json.load(f)
        best_values = grid_search['best_config']
        search_values = grid_search['search']

    if os.path.exists(out_dir):
        print("out directory %s already exists. beware of overwriting contents"
              % out_dir)
    else:
        os.mkdir(out_dir)
    grid_search_keys = list(grid_search.keys())

    best_solve_config = config['solve'].copy()
    for key, value in best_values.items():
        best_solve_config[key] = value

    config_num = 1
    for key, values in search_values.items():
        for value in values:
            whole_config_copy = config.copy()
            best_config_copy = best_solve_config.copy()
            best_config_copy[key] = value
            whole_config_copy['solve'] = best_config_copy
            with open(os.path.join(out_dir, str(config_num)), 'w') as outfile:
                json.dump(whole_config_copy, outfile, indent=2)
            config_num += 1
