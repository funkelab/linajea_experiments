import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from linajea import load_config
import linajea.evaluation
import logging
import argparse

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
# logging.getLogger(
#         'daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_config', help="test config file")
    parser.add_argument('-vs', '--vald_setup', help="validation setup (eg setup11_simple)")
    parser.add_argument('-vr', '--vald_region', help="validation region (eg late_early)")
    parser.add_argument('-vf', '--vald_frames', help="validation frames",
                        nargs='+', type=int, default=None)
    parser.add_argument('--version', default='v1.3-dev')
    parser.add_argument('--db_name', default=None)
    args = parser.parse_args()

    config = load_config(args.eval_config)
    sample = config['general']['sample']
    setup = args.vald_setup
    region = args.vald_region
    if args.db_name is None:
        if config['general']['db_name']:
            db_name = config['general']['db_name']
        else:
            db_name = f"linajea_{sample}_{setup}_{region}_400000_te"
    else:
        db_name = args.db_name

    print(db_name)
    best_vald_config = linajea.evaluation.get_best_result(
            db_name,
            config['general']['db_host'],
            sample=config['general']['sample'],
            frames=args.vald_frames,
            filter_params={'version': args.version})
