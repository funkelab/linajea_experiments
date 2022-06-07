import logging
from linajea.tracking import greedy_track
import argparse
from daisy import Roi

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

logging.getLogger('linajea.tracking.greedy_track').setLevel(logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--db_host',
            default="mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin")
    parser.add_argument('-db', '--db_name', required=True)
    parser.add_argument('-k', '--selected_key', required=True)
    parser.add_argument('-t', '--threshold', default=0)
    parser.add_argument('-o', '--offset', type=int, nargs=4)
    parser.add_argument('-s', '--shape', type=int, nargs=4)
    parser.add_argument('-n', '--new-seeds', action='store_true')
    args = parser.parse_args()
    if args.offset is not None:
        roi = Roi(args.offset, args.shape)
    else:
        roi = None
    logging.info("Running greedy tracking")
    greedy_track(
        args.db_name,
        args.db_host,
        args.selected_key,
        args.threshold,
        roi=roi,
        allow_new_tracks=args.new_seeds)
