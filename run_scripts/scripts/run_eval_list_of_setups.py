from __future__ import absolute_import
import funlib.run as funlib
import os
from linajea import load_config
from linajea_utils import write_grid_search_configs
import logging
import subprocess
import time
import bmonitor
import argparse
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# logger.setLevel(logging.DEBUG)

setups = [
    "setup211_simple_vald_side_1",
    "setup211_simple_vald_side_2"
#    "setup111_simple_eval_side_1"
#    "setup111_simple_eval_side_2"
#    "setup09_simple_early_middle",
#    "setup09_shift_early_middle",
#    "setup09_divisions_early_middle",
#    "setup09_shift_divisions_early_middle",
#    "setup11_simple_early_middle",
#    "setup11_shift_early_middle",
#    "setup11_divisions_early_middle",
#    "setup11_shift_divisions_early_middle",
#    "setup09_simple_late_middle",
#    "setup09_shift_late_middle",
#    "setup09_divisions_late_middle",
#    "setup09_shift_divisions_late_middle",
#    "setup11_simple_late_middle",
#    "setup11_shift_late_middle",
#    "setup11_divisions_late_middle",
#    "setup11_shift_divisions_late_middle",
#    "setup09_simple_middle_early",
#    "setup09_shift_middle_early",
#    "setup09_divisions_middle_early",
#    "setup09_shift_divisions_middle_early",
#    "setup11_simple_middle_early",
#    "setup11_shift_middle_early",
#    "setup11_divisions_middle_early",
#    "setup11_shift_divisions_middle_early",
#    "setup09_simple_middle_late",
#    "setup09_shift_middle_late",
#    "setup09_divisions_middle_late",
#    "setup09_shift_divisions_middle_late",
#    "setup11_simple_middle_late",
#    "setup11_shift_middle_late",
#    "setup11_divisions_middle_late",
#    "setup11_shift_divisions_middle_late",
#    "setup09_simple_early_late",
#    "setup09_shift_early_late",
#    "setup09_divisions_early_late",
#    "setup09_shift_divisions_early_late",
#    "setup11_simple_early_late",
#    "setup11_shift_early_late",
#    "setup11_divisions_early_late",
#    "setup11_shift_divisions_early_late",
#    "setup09_simple_late_early",
#    "setup09_shift_late_early",
#    "setup09_divisions_late_early",
#    "setup09_shift_divisions_late_early",
#    "setup11_simple_late_early",
#    "setup11_shift_late_early",
#    "setup11_divisions_late_early",
#    "setup11_shift_divisions_late_early",
]


def bkill(jobid):
    bkill_cmd = 'bkill %s' % jobid
    subprocess.run(bkill_cmd, shell=True)


def retry_failed(jobid, num_configs):
    retry_exited_cmd = 'brequeue -J "retry_gs[1-%d]" -e %s' % (num_configs,
                                                               jobid)
    subprocess.run(retry_exited_cmd)
    bwait_cmd = "bwait -w 'ended(%s)' -t 240"
    subprocess.check_call(bwait_cmd)


def check_job_failure_loop(retries, jobid, num_configs):
    if not bmonitor.is_ended(jobid, array=False):
        logger.warn("Bjobs not yet reporting completion."
                    " Trying again in 5 seconds")
        time.sleep(5)
    assert bmonitor.is_ended(jobid, array=False)
    if bmonitor.is_done(jobid, array=False):
        return 0

    if retries == 0:
        raise RuntimeError("Job %s failed - see log" % job_name)

    new_jobid = retry_failed(jobid, num_configs)
    logger.debug("Old jobid: %s. New jobid: %s" % (jobid, new_jobid))
    check_job_failure_loop(retries - 1, new_jobid, num_configs)


def check_job_failure(jobid, array=True):
    if not bmonitor.is_ended(jobid, array=array):
        logger.warn("Bjobs not yet reporting completion."
                    " Trying again in 5 seconds")
        time.sleep(5)
    assert bmonitor.is_ended(jobid, array=array)
    if bmonitor.is_done(jobid, array=array):
        logger.info("Job %s completed successfully" % jobid)
    else:
        if array:
            job_status = bmonitor.get_array_jobs_status(jobid)
            logger.info("THese jobs had status exit: %s" % str(job_status['EXIT']))
        raise RuntimeError("Job %s failed - see log" % jobid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry_run', action='store_true')
    parser.add_argument('-r', '--retries', type=int, default=1)
    parser.add_argument('-v', '--validate', action='store_true')
    parser.add_argument('-g', '--grid_search_params', required=True)
    args = parser.parse_args()

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    grid_search_params = args.grid_search_params

    for setup in setups:
        logger.info("Processing setup %s" % setup)
        config_dir = "gs_" + setup
        sample_file = "config_files/%s.toml" % setup
        # write config files
        num_configs = write_grid_search_configs(
                sample_file,
                config_dir,
                grid_search_params,
                ignore_existing=True)
        logger.debug("Done writing grid search configs for %s" % setup)
        jobids = []
        for config in os.listdir(config_dir):
            config_path = os.path.join(config_dir, config)
            cmd = ["python 04_evaluate.py",
                   config_path]
            cmd = ' '.join(cmd)
            job_name = "eval_" + setup + '_' + config
            log_file = os.path.join("logs", '.'.join(
                [job_name, "%J", "summary"]))
            error_file = os.path.join("logs", '.'.join(
                [job_name, "%J", "error"]))
            first_config = load_config(os.path.join(config_dir, config))
            num_cpus = 8
            time_limit = 5  # hours
            time_limit = time_limit * 60  # minutes
            singularity_image = first_config['general']['singularity_image']
            singularity_image = os.path.join('/nrs/funke/singularity',
                                             singularity_image + '.img')
            queue = "normal"
            mount_dirs = ['/groups', '/nrs/']
            logger.debug("Constructing command")
            execute = not args.dry_run
            jobid = funlib.run(
                    command=cmd,
                    num_cpus=num_cpus,
                    num_gpus=0,
                    working_dir='.',
                    singularity_image=singularity_image,
                    queue=queue,
                    batch=True,
                    mount_dirs=mount_dirs,
                    execute=execute,
                    expand=True,
                    job_name=job_name,
                    log_file=log_file,
                    error_file=error_file,
                    flags=[  # '-R "select[broadwell]"',
                           # '-Q "all ~0"',
                           '-W %d' % time_limit,
                           '-P funke'])
            if args.dry_run:
                print(jobid)
                sys.exit(0)
            logger.debug("Job ID: %s" % jobid)
            jobids.append(jobid)
        for jobid in jobids:
            bmonitor.wait_for_job_end(jobid)
            # Check for errors
            check_job_failure(jobid, array=False)
