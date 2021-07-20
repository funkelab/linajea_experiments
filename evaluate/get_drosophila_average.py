import linajea.evaluation as lev
import csv
from graph_from_csv import graph_from_csv
import argparse
import logging

logger = logging.getLogger(__name__)


def write_segments(outfile, results_list):
    print(results_list[0])
    segments_list = [r['correct_segments'] for r in results_list]
    max_seg_length = max([int(k) for k in segments_list[0].keys()])
    percents_list = []
    for segments in segments_list:
        percents = {}
        for i in range(1, max_seg_length + 1):
            correct, total = segments[str(i)]
            if total == 0:
                percent = 0.0
            else:
                percent = correct / total
            percents[str(i)] = percent
        percents_list.append(percents)
    print(percents_list[0])

    with open(outfile, 'w') as f:
        seg_writer = csv.DictWriter(
                f,
                fieldnames=[
                    'time', 'percent'],
                extrasaction='ignore')
        seg_writer.writeheader()
        for i in range(1, max_seg_length + 1):
            print('length of percents list:')
            print(len(percents_list))
            average = sum(p[str(i)] for p in percents_list) / len(percents_list)
            row = {'time': i, 'percent': average}
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
    our_results = []
    greedy_results = []
    tgmm_results = []
    out_file = "drosophila_average.csv"

    for region in regions:
            print("getting best result for region %s" % region)
            db_name = "linajea_120828_setup211_simple_%s_400000_te" % region
            best_dict = lev.get_best_result(db_name, db_host,
                    filter_params=filter_params)
            print("Best key for region %s: %d" % (region, best_dict['_id']))
            best_dict['region'] = region
            best_dict['model'] = 'ours'
            best_dict['sum'] = best_dict['sum_errors']
            best_dict['fn_edges'] = best_dict['fn_edges'] / best_dict['gt_edges']
            best_dict['identity_switches'] = best_dict['identity_switches'] / best_dict['gt_edges']
            best_dict['fp_divisions'] = best_dict['fp_divisions'] / best_dict['gt_edges']
            best_dict['fn_divisions'] = best_dict['fn_divisions'] / best_dict['gt_edges']
            best_dict['sum'] = best_dict['sum'] / best_dict['gt_edges']
            our_results.append(best_dict)

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
            greedy_result['fn_edges'] = greedy_result['fn_edges'] / greedy_result['gt_edges']
            greedy_result['identity_switches'] = greedy_result['identity_switches'] / greedy_result['gt_edges']
            greedy_result['fp_divisions'] = greedy_result['fp_divisions'] / greedy_result['gt_edges']
            greedy_result['fn_divisions'] = greedy_result['fn_divisions'] / greedy_result['gt_edges']
            greedy_result['sum'] = greedy_result['sum'] / greedy_result['gt_edges']
            greedy_results.append(greedy_result)

            tgmm_db = 'linajea_120828_tgmm_%s' % region
            tgmm_dict = lev.get_best_tgmm_result(tgmm_db, db_host)
            tgmm_dict['region'] = region
            tgmm_dict['model'] = "TGMM"
            tgmm_dict['sum'] = tgmm_dict['sum_errors']
            tgmm_dict['fn_edges'] = tgmm_dict['fn_edges'] / tgmm_dict['gt_edges']
            tgmm_dict['identity_switches'] = tgmm_dict['identity_switches'] / tgmm_dict['gt_edges']
            tgmm_dict['fp_divisions'] = tgmm_dict['fp_divisions'] / tgmm_dict['gt_edges']
            tgmm_dict['fn_divisions'] = tgmm_dict['fn_divisions'] / tgmm_dict['gt_edges']
            tgmm_dict['sum'] = tgmm_dict['sum'] / tgmm_dict['gt_edges']
            tgmm_results.append(tgmm_dict)

    def get_average(results, name):
        average = {'model': name}
        average['fn_edges'] = sum([r['fn_edges'] for r in results]) / len(results)
        average['identity_switches'] = sum([r['identity_switches'] for r in results]) / len(results)
        average['fp_divisions'] = sum([r['fp_divisions'] for r in results]) / len(results)
        average['fn_divisions'] = sum([r['fn_divisions'] for r in results]) / len(results)
        average['sum'] = sum([r['sum'] for r in results]) / len(results)
        return average

    with open(out_file, 'w') as f:
        writer = csv.DictWriter(
                f,
                fieldnames=[
                    'model', 'fn_edges',
                    'identity_switches', 'fp_divisions', 'fn_divisions',
                    'sum'],
                extrasaction='ignore')
        writer.writeheader()
        writer.writerow(get_average(our_results, 'ours'))
        write_segments('dros_segments_averaged_ours.csv', our_results)
        writer.writerow(get_average(greedy_results, 'greedy'))
        write_segments('dros_segments_averaged_greedy.csv', greedy_results)
        writer.writerow(get_average(tgmm_results, 'TGMM'))
        write_segments('dros_segments_averaged_tgmm.csv', tgmm_results)
