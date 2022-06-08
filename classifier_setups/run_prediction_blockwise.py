import argparse
import daisy
from funlib.run import run
from linajea import CandidateDatabase
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')


def predict_worker(block, config_file, db_name, db_host):
    queue = 'gpu_rtx'
    image = '/nrs/funke/singularity/linajea/linajea:v1.3-dev.img'
    offset = block.read_roi.get_offset()
    shape = block.read_roi.get_shape()
    cmd = run(
            command='python -u %s %s -db %s -ro %d %d %d %d -rs %d %d %d %d -o'
                    % ('predict.py',
                        config_file,
                        db_name,
                        offset[0],
                        offset[1],
                        offset[2],
                        offset[3],
                        shape[0],
                        shape[1],
                        shape[2],
                        shape[3],
                       ),
            queue=queue,
            num_gpus=1,
            num_cpus=5,
            singularity_image=image,
            mount_dirs=['/groups', '/nrs'],
            execute=False,
            expand=False,
            flags=[' -P funke ']
            )
    logger.info("Starting predict worker...")
    logger.info("Command: %s" % str(cmd))
    worker_id = daisy.Context.from_env().worker_id
    worker_time = time.time()
    daisy.call(
        cmd,
        log_out='logs/predict_vgg_%d_%d.out' % (worker_time, worker_id),
        log_err='logs/predict_vgg_%d_%d.err' % (worker_time, worker_id))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-db', '--db_name', required=True)
    args = parser.parse_args()
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"

    cand_db = CandidateDatabase(args.db_name, db_host, 'r')
    total_roi = cand_db.get_nodes_roi()
    block_roi = daisy.Roi((0, 0, 0, 0), (5, 500, 500, 500))

    success = daisy.run_blockwise(
            total_roi,
            block_roi,
            block_roi,
            process_function=lambda b: predict_worker(
                b,
                args.config,
                args.db_name,
                db_host),
            num_workers=8,
            read_write_conflict=False,
            max_retries=0,
            fit='overhang')
