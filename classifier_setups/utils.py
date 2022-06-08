import numpy as np
import logging
import os
from match_candidate_nodes_to_gt import match_candidates, write_results

logger = logging.getLogger(__name__)


def load_cand_locations(filename, return_type='dict'):
    '''dataset options: train, validate
       return type options: dict, list'''
    if return_type == 'dict':
        locations = {}
    elif return_type == 'list':
        locations = []
        labels = []
        label_dict = {'division': 0, 'child': 1, 'continuation': 2}
    else:
        raise ValueError("Return type must be dict or list, got %s",
                         return_type)
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(' ')
            label = tokens[0]
            location = tokens[1:]
            location = np.array([int(x) for x in location])
            if return_type == 'dict':
                if label not in locations:
                    locations[label] = []
                locations[label].append(location)
            elif return_type == 'list':
                locations.append(location)
                labels.append(label_dict[label])
    if return_type == 'dict':
        return locations
    elif return_type == 'list':
        return labels, locations


def get_cell_cycle_labels(config, dataset='train', return_type='dict'):
    '''dataset options: train, validate
       return type options: dict, list'''
    filename = config['setup_dir'] + '/labels/%s.txt' % dataset
    if not os.path.exists(filename):
        logger.info("Generating %s labels" % dataset)
        dataset_config = config[dataset]
        label_to_points = match_candidates(
                dataset_config['candidate_db'],
                dataset_config['gt_db'],
                config['db_host'],
                dataset_config['frames'],
                dataset_config['match_distance'],
                exclude=dataset_config.get('exclude', None))
        write_results(label_to_points, filename)
        if return_type == 'dict':
            return label_to_points
        elif return_type == 'list':
            positions = []
            labels = []
            label_dict = {'division': 0, 'child': 1, 'continuation': 2}
            for label, points in label_to_points.items():
                for point in points:
                    labels.append(label_dict[label])
                    positions.append(point)

            return labels, positions
        else:
            raise ValueError("Invalid return type %s: must be dict or list",
                             return_type)
    else:
        logger.info("Reading %s labels from %s", dataset, filename)
        return load_cand_locations(filename, return_type=return_type)
