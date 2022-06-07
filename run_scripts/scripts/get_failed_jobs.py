import argparse
import bmonitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--jobid', help="job id")
    args = parser.parse_args()

    print(bmonitor.get_array_jobs_status(args.jobid))
