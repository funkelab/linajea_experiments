import gunpowder as gp
import logging
import numpy as np
import os
import argparse
from linajea.gunpowder import WriteCells
from linajea.process_blockwise import write_done
from linajea import load_config

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logging.getLogger('gunpowder.nodes.zarr_source').setLevel(logging.DEBUG)
logging.getLogger('gunpowder.nodes.hdf5like_source_base')\
    .setLevel(logging.DEBUG)


def predict(
        setup,
        iteration,
        sample,
        data_file,
        db_host,
        db_name,
        score_threshold,
        data_dir='../01_data',
        out_file=None,
        processes_per_worker=5):

    setups_dir = os.path.dirname(os.path.realpath(__file__))
    setup_dir = os.path.join(setups_dir, setup)
    config = load_config(os.path.join(setup_dir, 'test_net_config.json'))
    # get absolute paths
    if os.path.isfile(sample) or sample.endswith((".zarr", ".n5")):
        sample_dir = os.path.abspath(os.path.join(data_dir,
                                                  os.path.dirname(sample)))
    else:
        sample_dir = os.path.abspath(os.path.join(data_dir, sample))
    attributes = load_config(os.path.join(sample_dir, 'attributes.json'))

    voxel_size = attributes['resolution']

    raw = gp.ArrayKey('RAW')
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')

    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(config['input_shape'])*voxel_size
    output_size = gp.Coordinate(config['output_shape_2'])*voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(parent_vectors, output_size)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)

    if data_file.endswith(".zarr") or data_file.endswith(".n5"):
        logging.info(
            os.path.join(
                data_dir,
                str(sample),
                data_file))
        source = gp.ZarrSource(
            os.path.join(
                data_dir,
                str(sample),
                data_file),
            {raw: 'raw'},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)})
    elif data_file.endswith(".klb"):
        source = gp.KlbSource(
                os.path.join(
                    data_dir,
                    str(sample),
                    data_file),
                raw,
                gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size))
    else:
        raise NotImplementedError("Data must end with klb, zarr, or n5, got %s"
                                  % data_file)

    with gp.build(source):
        raw_spec = source.spec[raw]

    pipeline = (
        source +
        gp.Pad(raw, size=None) +
        gp.Normalize(raw) +
        gp.tensorflow.Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d' % iteration),
            graph=os.path.join(setup_dir, 'test_net.meta'),
            inputs={
                config['raw']: raw,
            },
            outputs={
                config['parent_vectors_cropped']: parent_vectors,
                config['cell_indicator_cropped']: cell_indicator,
                config['maxima']: maxima,
            },
            array_specs={
                parent_vectors: gp.ArraySpec(
                    roi=None,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.float32
                ),
                cell_indicator: gp.ArraySpec(
                    roi=None,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.float32
                ),
                maxima: gp.ArraySpec(
                    roi=None,
                    voxel_size=raw_spec.voxel_size,
                    dtype=np.uint8
                )
            },
            skip_empty=True
        ))
    if out_file:
        pipeline = (
            pipeline +

            gp.ZarrWrite(
                dataset_names={
                    parent_vectors: 'volumes/parent_vectors',
                    cell_indicator: 'volumes/cell_indicator'
                    },
                output_filename=os.path.join(setup_dir, out_file)))

    pipeline = (
        pipeline +

        WriteCells(
            maxima,
            cell_indicator,
            parent_vectors,
            score_threshold=score_threshold,
            db_host=db_host,
            db_name=db_name) +

        gp.PrintProfilingStats(every=10) +
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                parent_vectors: 'write_roi',
                cell_indicator: 'write_roi',
                maxima: 'write_roi'
            },
            num_workers=processes_per_worker,
            block_done_callback=lambda b, st, et: write_done(
                b,
                'predict',
                db_name,
                db_host)
            )
    )

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    parser.add_argument('--iteration', type=int,
                        help='which saved model to use for prediction')
    args = parser.parse_args()

    config = load_config(args.config)

    iteration = args.iteration
    setup = config['general']['setup']
    sample = config['general']['sample']
    db_host = config['general']['db_host']
    db_name = config['general']['db_name']
    score_threshold = config['predict']['cell_score_threshold']
    data_dir = config['general']['data_dir']\
        if 'data_dir' in config['general'] else '../data'
    data_file = config['general']['data_file']\
        if 'data_file' in config['general'] else sample + '.n5'
    prediction_zarr_file = config['predict']['output_zarr']\
        if 'output_zarr' in config['predict'] else None
    processes_per_worker = config['predict']['processes_per_worker']\
        if 'processes_per_worker' in config['predict'] else 5

    predict(
        setup,
        iteration,
        sample,
        data_file,
        db_host,
        db_name,
        score_threshold,
        data_dir=data_dir,
        out_file=prediction_zarr_file,
        processes_per_worker=processes_per_worker)
