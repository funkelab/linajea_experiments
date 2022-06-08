import argparse
import gunpowder as gp
from gunpowder.torch import Train
import logging
import math
import numpy as np
import os
import toml
import torch
from utils import get_cell_cycle_labels
from vgg_model import get_cell_cycle_model
from predict import predict
from evaluate import evaluate
from position_dataset import PositionDataset
from data_loader import get_data_loader
from daisy import Coordinate
import time

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


class AddGtLabel(gp.BatchFilter):

    def __init__(self, array, label):

        self.array = array
        self.label = label

    def setup(self):

        self.provides(self.array, gp.ArraySpec(nonspatial=True))

    def prepare(self, request):
        pass

    def process(self, batch, request):
        batch.arrays[self.array] = \
            gp.Array(np.array(self.label, dtype=np.int64),
                     gp.ArraySpec(nonspatial=True))


def train_until(
        model,
        iterations,
        setup_dir,
        data_file,
        candidate_locations,
        batch_size,
        snapshot_frequency,
        validation_frequency,
        learning_rate,
        num_workers=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    print("Optimizer", type(optimizer))
    raw = gp.ArrayKey('RAW')
    # pred_labels = gp.ArrayKey('PRED_LABELS')
    pred_probs = gp.ArrayKey('PRED_PROBS')
    gt_labels = gp.ArrayKey('GT_LABELS')

    if data_file.endswith(("h5", "hdf")):
        SourceNode = gp.Hdf5Source
    elif data_file.endswith("zarr") or data_file.endswith("n5"):
        SourceNode = gp.ZarrSource
    else:
        raise RuntimeError("Invalid input format %s", data_file)

    source = SourceNode(
        data_file,
        datasets={
            raw: 'raw',
        },
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
        }
    )

    with gp.build(source):
        voxel_size = source.spec[raw].voxel_size
    logger.info("voxel_size: %s", str(voxel_size))
    input_shape = config['input_shape']
    input_shape_world = gp.Coordinate(input_shape)*voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_shape_world)
    request[gt_labels] = gp.ArraySpec(nonspatial=True)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_shape_world)
    # snapshot_request[pred_labels] = gp.ArraySpec(nonspatial=True)
    snapshot_request[pred_probs] = gp.ArraySpec(nonspatial=True)
    locations_by_class = {
        0: candidate_locations['division'],
        1: candidate_locations['child'],
        2: candidate_locations['continuation'],
    }

    sources = tuple(
        SourceNode(
            data_file,
            datasets={
                raw: 'raw',
            },
            array_specs={
                raw: gp.ArraySpec(interpolatable=True),
            }
        ) +
        gp.Pad(raw, gp.Coordinate((1, 30, 30, 30))) +
        gp.Normalize(raw) +
        gp.SpecifiedLocation(
            locations_by_class[cls],
            choose_randomly=True) +
        AddGtLabel(gt_labels, cls)
        for cls in range(3))

    pipeline = (
        sources +
        gp.RandomProvider() +

        # elastically deform the batch
        gp.ElasticAugment(
            control_point_spacing=(4, 8, 8),
            jitter_sigma=(1, 1, 1),
            rotation_interval=[0, math.pi/2.0],
            subsample=4) +

        # apply transpose and mirror augmentations
        gp.SimpleAugment(mirror_only=[1, 2, 3], transpose_only=[2, 3]) +

        # scale and shift the intensity of the raw array
        gp.IntensityAugment(raw, 0.9, 1.1, -0.001, 0.001) +

        gp.PreCache(
            cache_size=40,
            num_workers=num_workers) +

        gp.Stack(batch_size) +

        Train(
            model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            inputs={
                'raw': raw,
            },
            loss_inputs={
                0: pred_probs,
                1: gt_labels,
            },
            outputs={
                0: pred_probs,
                # 'pred_probs': pred_probs,
            },
            array_specs={
                # pred_labels: gp.ArraySpec(nonspatial=True),
                pred_probs: gp.ArraySpec(nonspatial=True)
            },
            log_dir=setup_dir,
            log_every=100,
            save_every=25000,
            checkpoint_basename=os.path.join(setup_dir, "model")) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot({
                raw: 'volumes/raw',
                gt_labels: 'gt_labels',
                # pred_labels: 'pred_labels',
                pred_probs: 'pred_probs'
            },
            output_dir=os.path.join(setup_dir, 'snapshots'),
            output_filename='batch_{iteration}.hdf',
            every=snapshot_frequency,
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every few iterations
        gp.PrintProfilingStats(every=10)
    )

    logger.info("Starting training...")
    vald_labels, vald_positions = get_cell_cycle_labels(
            config, dataset='validate', return_type='list')
    roi_size = Coordinate(config['input_shape'])
    position_dataset = PositionDataset(
            vald_positions,
            config['data_file'],
            roi_size)
    vald_data_loader = get_data_loader(
            position_dataset,
            config['batch_size'],
            roi_size,
            num_workers=num_workers)
    vald_iterations = []
    scores = []
    matrix_file = os.path.join(setup_dir, "validation_matrices.csv")
    best_score = load_previous_vald_best(matrix_file)
    start_time = time.time()
    with gp.build(pipeline):
        while True:
            batch = pipeline.request_batch(request)
            if batch.iteration % validation_frequency == 0:
                logger.info("predicting validation set with iteration %d",
                            batch.iteration)
                matrix, score = run_and_save_validation(
                        model, batch.iteration,
                        vald_data_loader, vald_labels,
                        matrix_file)
                scores.append(score)
                vald_iterations.append(batch.iteration)
                if score > best_score:
                    logger.info("Best score %f at iteration %d:"
                                " saving checkpoint", score, batch.iteration)
                    best_score = score
                    checkname = os.path.join(
                            config['setup_dir'],
                            'model_checkpoint_%d' % batch.iteration)
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict":
                                    optimizer.state_dict()},
                               checkname)
            if batch.iteration >= iterations:
                logger.info("saving final iteration checkpoint")
                checkname = os.path.join(
                        config['setup_dir'],
                        'model_checkpoint_%d' % batch.iteration)
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict":
                                optimizer.state_dict()},
                           checkname)
                break
    training_time = time.time() - start_time
    logger.info("Training finished in %d seconds", training_time)
    logger.info("Iterations: %s", str(vald_iterations))
    logger.info("Scores: %s", str(scores))


def run_and_save_validation(
        model,
        iteration,
        data_loader,
        labels,
        matrix_file):
    ids, probabilities = predict(model, data_loader)
    predicted_labels = [p.index(max(p)) for p in probabilities]
    predicted_labels = predicted_labels[0:len(labels)]
    confusion_matrix, score = evaluate(labels,
                                       predicted_labels)
    with open(matrix_file, 'a') as f:
        f.write(str(iteration) + ',')
        to_write = confusion_matrix.flatten()
        f.write(','.join(list(map(str, to_write))))
        f.write(',' + str(score))
        f.write('\n')
    logger.debug("Validation score %f at iteration %d",
                 score, iteration)
    return confusion_matrix, score


def load_previous_vald_best(matrix_file):
    best_score = 0
    if os.path.exists(matrix_file):
        logger.debug("Loading previous best score from %s", matrix_file)
        with open(matrix_file, 'r') as f:
            for line in f.readlines():
                score = line.strip().split(',')[-1]
                score = float(score)
                if score > best_score:
                    best_score = score
    logger.debug("Starting best score: %f", best_score)
    return best_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file')
    parser.add_argument('-i', '--iterations', type=int,
                        help='number ofiterations to train',
                        default=100000)
    parser.add_argument('-s', '--snap-frequency', type=int,
                        help='number of iterations to save snapshots',
                        default=1000)
    parser.add_argument('-v', '--validation-frequency', type=int,
                        help='number of iterations to validate and save best'
                             'checkpoint',
                        default=2000)
    args = parser.parse_args()
    with open(args.config) as f:
        config = toml.load(f)
    logger.info("Config: %s", config)
    training_labels = get_cell_cycle_labels(
            config, dataset='train', return_type='dict')
    model = get_cell_cycle_model(args.config)
    train_until(
        model,
        args.iterations,
        config['setup_dir'],
        config['data_file'],
        training_labels,
        config['batch_size'],
        args.snap_frequency,
        args.validation_frequency,
        learning_rate=config['lr'])
