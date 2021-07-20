import linajea.evaluation as lev
import csv
import argparse
import logging

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
            percent = correct / total
            row = {'time': i, 'correct': correct,
                   'total': total, 'percent': percent}
            seg_writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--normalize', action='store_true')
    args = parser.parse_args()
    # need tgmm, our best result on side 1 and side 2
    setup = 'setup211_simple'
    regions = ['eval_side_1', 'eval_side_2']
    sample = '120828'
    db_host = "localhost"  # TODO: Replace with MongoDB URL
    filter_params = {'version': "v1.3-dev"}
    for region in regions:
        out_file = "drosophila_" + region + ".csv"
        segments_file = "drosophila_segments_" + region + "_%s.csv"
        with open(out_file, 'w') as f:
            # side, model, fn, is, fp-d, fn-d, sum
            writer = csv.DictWriter(
                    f,
                    fieldnames=['model', 'gt_edges', 'fn_edges',
                        'identity_switches', 'fp_divisions', 'fn_divisions',
                        'sum', 'aeftl'],
                    extrasaction='ignore')
            writer.writeheader()

            print("getting best result for region %s" % region)
            db_name = "linajea_120828_setup211_simple_%s_400000_te" % region
            best_dict = lev.get_best_result(db_name, db_host,
                    filter_params=filter_params)
            print("Best key for region %s: %d" % (region, best_dict['_id']))
            best_dict['region'] = region
            best_dict['model'] = 'ours'
            best_dict['sum'] = best_dict['sum_errors']
            if args.normalize:
                best_dict['fn_edges'] = best_dict['fn_edges'] / best_dict['gt_edges']
                best_dict['identity_switches'] = best_dict['identity_switches'] / best_dict['gt_edges']
                best_dict['fp_divisions'] = best_dict['fp_divisions'] / best_dict['gt_edges']
                best_dict['fn_divisions'] = best_dict['fn_divisions'] / best_dict['gt_edges']
                best_dict['sum'] = best_dict['sum'] / best_dict['gt_edges']
            writer.writerow(best_dict)
            write_segments(segments_file % 'ours', best_dict['correct_segments'])

            # Get greedy tracking result for this region
            greedy_result = lev.get_greedy(
                    db_name,
                    db_host)
            greedy_result['model'] = 'greedy'
            greedy_result['sum'] = (
                    greedy_result['fn_edges'] +
                    greedy_result['identity_switches'] +
                    greedy_result['fp_divisions'] +
                    greedy_result['fn_divisions'])
            logger.debug("Greedy result: %s" % greedy_result)
            if args.normalize:
                greedy_result['fn_edges'] = greedy_result['fn_edges'] / greedy_result['gt_edges']
                greedy_result['identity_switches'] = greedy_result['identity_switches'] / greedy_result['gt_edges']
                greedy_result['fp_divisions'] = greedy_result['fp_divisions'] / greedy_result['gt_edges']
                greedy_result['fn_divisions'] = greedy_result['fn_divisions'] / greedy_result['gt_edges']
                greedy_result['sum'] = greedy_result['sum'] / greedy_result['gt_edges']
            writer.writerow(greedy_result)
            write_segments(segments_file % 'greedy', greedy_result['correct_segments'])
            tgmm_db_name = 'linajea_120828_tgmm_' + region
            tgmm_dict = lev.get_best_tgmm_result(tgmm_db_name, db_host)
            tgmm_dict['region'] = region
            tgmm_dict['model'] = "TGMM"
            tgmm_dict['sum'] = tgmm_dict['sum_errors']
            if args.normalize:
                tgmm_dict['fn_edges'] = tgmm_dict['fn_edges'] / tgmm_dict['gt_edges']
                tgmm_dict['identity_switches'] = tgmm_dict['identity_switches'] / tgmm_dict['gt_edges']
                tgmm_dict['fp_divisions'] = tgmm_dict['fp_divisions'] / tgmm_dict['gt_edges']
                tgmm_dict['fn_divisions'] = tgmm_dict['fn_divisions'] / tgmm_dict['gt_edges']
                tgmm_dict['sum'] = tgmm_dict['sum'] / tgmm_dict['gt_edges']
            writer.writerow(tgmm_dict)
            write_segments(segments_file % 'tgmm', tgmm_dict['correct_segments'])
