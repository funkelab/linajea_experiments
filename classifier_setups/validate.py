import argparse
import logging
import os
import toml
import torch
from daisy import Coordinate
from vgg_model import get_cell_cycle_model
from position_dataset import get_data_loader
from evaluate import evaluate
from predict import predict
from utils import get_cell_cycle_labels

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def run_validation(config_file, num_workers, iteration, from_scratch=False):
    with open(config_file) as f:
        config = toml.load(f)
    logger.info("Config: %s", config)

    results_file = os.path.join(
            config['setup_dir'],
            'validation_results_%d.txt' % iteration)
    pred_file = os.path.join(
            config['setup_dir'],
            'validation_prediction_%d.txt' % iteration)

    if os.path.exists(results_file) and not from_scratch:
        logger.info("Validation results already exist: returning")
        return
    if os.path.exists(pred_file) and not from_scratch:
        logger.info("Validation predictions already exist: "
                    "skipping prediction")
        true_labels = []
        pred_labels = []
        with open(pred_file, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                true_labels.append(tokens[0])
                pred_labels.append(tokens[1])
        confusion_matrix, score = evaluate(true_labels, pred_labels)
        with open(results_file, 'w') as f:
            f.write(str(confusion_matrix))
            f.write(str(score))
        logger.info("Score: %s", str(score))
        return score

    vald_labels, vald_positions = get_cell_cycle_labels(
            config, dataset='validate', return_type='list')

    model = get_cell_cycle_model(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkname = os.path.join(config['setup_dir'],
                             'model_checkpoint_%d' % iteration)
    checkpoint = torch.load(checkname, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    data_loader = get_data_loader(
            vald_positions,
            config['data_file'],
            Coordinate(config['input_shape']),
            config['batch_size'],
            args.num_workers,
            args.prefetch)
    logger.info("predicting validation set with iteration %d", iteration)
    predicted_probs = predict(model, data_loader)
    predicted_labels = [p.index(max(p)) for p in predicted_probs]
    with open(pred_file, 'w') as f:
        for true, pred, probs in zip(
                vald_labels, predicted_labels, predicted_probs):
            f.write(' '.join(map(str, [true, pred, probs])))
            f.write('\n')
    logger.info("evaluating validation set with iteration %d", iteration)
    confusion_matrix, score = evaluate(vald_labels, predicted_labels)
    with open(results_file, 'w') as f:
        f.write(str(confusion_matrix))
        f.write(str(score))
    logger.info("Score: %s", str(score))
    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('-n', '--num-workers', type=int, default=5,
                        help='number of cpu workers for prefetching data')
    parser.add_argument('-i', '--iterations', type=int, default=100000,
                        help='which checkpoint to use for validation')
    parser.add_argument('-p', '--prefetch', type=int, default=30,
                        help='size of prefetch queue?')
    args = parser.parse_args()
    run_validation(args.config, args.num_workers, args.iterations)
