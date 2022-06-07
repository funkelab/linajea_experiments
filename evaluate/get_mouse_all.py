import linajea.evaluation as lev
import csv
# from graph_from_csv import graph_from_csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_segments(outfile, segments):
    with open(outfile, 'w') as f:
        seg_writer = csv.DictWriter(
                f,
                fieldnames=[
                    'time', 'correct', 'total', 'percent'],
                extrasaction='ignore')
        seg_writer.writeheader()
        max_seg_length = max([int(k) for k in segments.keys()])
        for i in range(1, max_seg_length + 1):
            correct, total = segments[str(i)]
            if total == 0:
                continue
            percent = correct / total
            row = {'time': i, 'correct': correct,
                   'total': total, 'percent': percent}
            seg_writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--normalize', action='store_true')
    args = parser.parse_args()
    filter_params = {'version': 'v1.3-dev'}
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    db_name = 'linajea_140521_setup11_simple_all_400000'

    out_file = "mouse_all.csv"
    segments_file = "mouse_segments_all.csv"
    results = lev.get_results(db_name, db_host)
    eval_result = results.iloc[0].copy()
    eval_result['model'] = 'ours'
    eval_result['sum'] = (
            eval_result['fn_edges'] +
            eval_result['identity_switches'] +
            eval_result['fp_divisions'] +
            eval_result['fn_divisions'])
    logger.debug("Evaluation result: %s" % eval_result)
    if args.normalize:
        eval_result['fn_edges'] = eval_result['fn_edges'] / eval_result['gt_edges']
        eval_result['identity_switches'] = eval_result['identity_switches'] / eval_result['gt_edges']
        eval_result['fp_divisions'] = eval_result['fp_divisions'] / eval_result['gt_edges']
        eval_result['fn_divisions'] = eval_result['fn_divisions'] / eval_result['gt_edges']
        eval_result['sum'] = eval_result['sum'] / eval_result['gt_edges']

    # side, model, fn, is, fp-d, fn-d, sum
    with open(out_file, 'w') as f:
        writer = csv.DictWriter(
                f,
                fieldnames=[
                    'model', 'gt_edges', 'fn_edges',
                    'identity_switches', 'fp_divisions', 'fn_divisions', 'sum'],
                extrasaction='ignore')
        writer.writeheader()
        writer.writerow(eval_result)
