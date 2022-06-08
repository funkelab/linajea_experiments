import daisy
from funlib.run import run
import logging
import time
from linajea import load_config, CandidateDatabase
import argparse
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def predict_worker(block, config_file, iteration, overwrite=False):
    start = block.read_roi.get_offset()[0]
    end = start + block.read_roi.get_shape()[0]
    command = 'python predict.py -c %s -i %d -f %d %d' %\
        (config_file, iteration, start, end)
    if overwrite:
        command = command + ' -o'
    queue = 'gpu_rtx'
    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    config = load_config(config_file)
    setup = config['setup_dir']

    cmd = run(
            command=command,
            queue=queue,
            num_gpus=1,
            num_cpus=5,
            execute=False,
            expand=False,
            flags=[' -P funke ']
            )
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    daisy.call(
        cmd,
        log_out='logs/predict_%s_%d_%d.out' % (setup, worker_time, worker_id),
        log_err='logs/predict_%s_%d_%d.err' % (setup, worker_time, worker_id))

    logger.info("Predict worker finished")


def predict_blockwise(config_file, iteration,
                      frames=None, overwrite=False):
    config = load_config(config_file)
    db_name = config['validate']['candidate_db']
    db_host = config['db_host']
    num_workers = config['validate']['num_workers']
    # get ROI of source
    candidate_db = CandidateDatabase(db_name, db_host, 'a')
    total_roi = candidate_db.get_nodes_roi()
    logger.info("Got total roi from candidate_db")
    if frames is not None:
        total_roi = total_roi.intersect(
                daisy.Roi((frames[0], None, None, None),
                          (frames[1] - frames[0], None, None, None)))

    # create read and write ROI
    write_roi_shape = (1,) + tuple(total_roi.get_shape()[1:])
    block_write_roi = daisy.Roi((0, 0, 0, 0), write_roi_shape)
    block_read_roi = block_write_roi
    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s" % total_roi)
    logger.info("Block read  ROI = %s" % block_read_roi)
    logger.info("Block write ROI = %s" % block_write_roi)
    logger.info("Output ROI      = %s" % total_roi)

    logger.info("Starting block-wise processing...")
    daisy.run_blockwise(
        total_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: predict_worker(
            b,
            config_file,
            iteration,
            overwrite=overwrite),
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=0,
        fit='overhang')

    logger.info("Done running prediction blockwise per frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file', required=True)
    parser.add_argument('-i', '--iteration', type=int, required=True)
    parser.add_argument('-f', '--frames', type=int, nargs=2, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    predict_blockwise(
            args.config, args.iteration,
            frames=args.frames,
            overwrite=args.overwrite)
