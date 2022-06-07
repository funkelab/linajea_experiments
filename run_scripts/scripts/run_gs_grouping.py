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
        'setup211_simple_vald_side_1',
#        'setup211_simple_vald_side_2',
]


def bkill(jobid):
    bkill_cmd = 'bkill %s' % jobid
    subprocess.run(bkill_cmd, shell=True)


def retry_failed(jobid):
    retry_exited_cmd = 'brequeue -J "retry_gs" -e %s' % jobid
    subprocess.run(retry_exited_cmd)
    bwait_cmd = "bwait -w 'ended(%s)' -t 240" % jobid
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


def check_job_failure(jobid):
    if not bmonitor.is_ended(jobid, array=False):
        logger.warn("Bjobs not yet reporting completion."
                    " Trying again in 5 seconds")
        time.sleep(5)
    assert bmonitor.is_ended(jobid, array=False)
    if bmonitor.is_done(jobid, array=False):
        logger.info("Job %s completed successfully" % jobid)
    else:
        logger.warn("Job %s failed" % jobid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry_run', action='store_true')
    parser.add_argument('-r', '--retries', type=int, default=1)
    parser.add_argument('-g', '--grid_search_params', required=True)
    parser.add_argument('-cd', '--config_dir',
                        help='directory to write config files to')
    parser.add_argument('-s', '--group_size', type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    grid_search_params = args.grid_search_params

    for setup in setups:
        logger.info("Processing setup %s" % setup)
        if args.config_dir:
            config_dir = args.config_dir
        else:
            config_dir = "gs_" + setup
        sample_file = "config_files/%s.toml" % setup
        # write config files
        num_configs = write_grid_search_configs(
                sample_file,
                config_dir,
                grid_search_params,
                ignore_existing=True,
                group_size=args.group_size)
        logger.debug("Done writing grid search configs for %s" % setup)

        config_directories = []
        if args.group_size:
            for subdir_name in os.listdir(config_dir):
                config_directories.append(
                        os.path.join(config_dir, subdir_name))
        else:
            config_directories.append(config_dir)

        # run one job per subdirectory
        # TODO: get all configs from file
        jobids = []
        for dir_name in config_directories:
            sample_config = os.path.join(dir_name, os.listdir(dir_name)[0])
            cmd = ["python 03_solve.py",
                   sample_config,
                   '-pd', dir_name,
                   '-e'
                   ]
            cmd = ' '.join(cmd)
            job_name = "gs_" + setup
            if args.group_size:
                job_name = job_name + '_' + dir_name
            log_file = os.path.join("logs", '.'.join(
                [job_name, "%J", "summary"]))
            error_file = os.path.join("logs", '.'.join(
                [job_name, "%J", "error"]))
            first_config = load_config(sample_config)
            num_cpus = first_config['solve']['num_workers']
            time_limit = 10  # hours
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
                           #  '-Q "all ~0"',
                           '-P funke '
                           '-W %d' % time_limit])
            if args.dry_run:
                print(jobid)
                sys.exit(0)
            logger.debug("Job ID: %s" % jobid)
            jobids.append(jobid)
        for jobid in jobids:
            bmonitor.wait_for_job_end(jobid)

            # Check for errors
            check_job_failure(jobid)
