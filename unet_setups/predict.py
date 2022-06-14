import gunpowder as gp
import logging
import numpy as np
import os
import argparse
from linajea.gunpowder import WriteCells
from linajea.process_blockwise import write_done
from linajea.config import TrackingConfig
from linajea import (load_config,
                     construct_zarr_filename)

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


def predict(config):
    setup_dir = config.general.setup_dir
    iteration = config.inference.checkpoint
    data_file = config.inference.data_source.datafile.filename
    db_host = config.general.db_host
    db_name = config.inference.data_source.db_name
    score_threshold = config.inference.cell_score_threshold
    processes_per_worker = config.predict.processes_per_worker

    net_config = load_config(os.path.join(config.general.setup_dir,
                                          'test_net_config.json'))
    # get absolute paths
    # if os.path.isfile(sample) or sample.endswith((".zarr", ".n5")):
    #    sample_dir = os.path.abspath(os.path.join(data_dir,
    #                                               os.path.dirname(sample)))
    # else:
    #    sample_dir = os.path.abspath(os.path.join(data_dir, sample))
    # attributes = load_config(os.path.join(sample_dir, 'attributes.json'))

    voxel_size = gp.Coordinate(config.inference.data_source.voxel_size)

    raw = gp.ArrayKey('RAW')
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')

    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape_2'])*voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(parent_vectors, output_size)
    chunk_request.add(cell_indicator, output_size)
    chunk_request.add(maxima, output_size)

    if data_file.endswith(".zarr") or data_file.endswith(".n5"):
        logging.info(data_file)
        source = gp.ZarrSource(
            data_file,
            {raw: 'raw'},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True,
                    voxel_size=voxel_size)})
    elif data_file.endswith(".klb"):
        source = gp.KlbSource(
                data_file,
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
                net_config['raw']: raw,
            },
            outputs={
                net_config['parent_vectors_cropped']: parent_vectors,
                net_config['cell_indicator_cropped']: cell_indicator,
                net_config['maxima']: maxima,
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
    if config.predict.write_to_zarr:
        out_file = construct_zarr_filename(config, data_file,
                                           config.inference.checkpoint)
        pipeline = (
            pipeline +

            gp.ZarrWrite(
                dataset_names={
                    parent_vectors: 'volumes/parent_vectors',
                    cell_indicator: 'volumes/cell_indicator'
                    },
                output_filename=out_file))

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
    args = parser.parse_args()

    config = TrackingConfig.from_file(args.config)
    predict(config)
