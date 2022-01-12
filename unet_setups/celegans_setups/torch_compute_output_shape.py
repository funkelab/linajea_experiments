import argparse
import json
import os

import torch

from linajea.config import TrackingConfig
import torch_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)

    model = torch_model.UnetModelWrapper(config)
    model.eval()

    input_shape_predict = config.model.predict_input_shape
    trial_run_predict = model.forward(torch.zeros(input_shape_predict,
                                                  dtype=torch.float32))
    _, _, trial_max_predict, _ = trial_run_predict
    output_shape_predict = trial_max_predict.size()
    net_config = {
        'input_shape': input_shape_predict,
        'output_shape_2': output_shape_predict
    }
    print(net_config)
    with open('test_net_config.json', 'w') as f:
        json.dump(net_config, f)
