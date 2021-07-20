import linajea.evaluation as lev
import csv
import argparse


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
    setup = 'setup111_simple'
    regions = ['eval_side_1', 'eval_side_2']
    sample = '160328'
    db_host = "localhost"  # TODO: Replace with MongoDB URL
    # need tgmm, our best result on side 1 and side 2
    # model, gt_edges, fn, is, fp-d, fn-d, sum
    fieldnames = ['model', 'gt_edges', 'fn_edges',
            'identity_switches', 'fp_divisions', 'fn_divisions', 'sum', 'aeftl']
    for region in regions:
        out_file = "zebrafish_" + region + ".csv"
        segments_file = "zebrafish_segments_" + region + "_%s.csv"
        with open(out_file, 'w') as f:
            writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction='ignore')
            writer.writeheader()

            print("getting best result for region %s" % region)
            db_name = "linajea_160328_setup111_simple_%s_400000_te" % region
            best_dict = lev.get_best_result(db_name, db_host)
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
            if args.normalize:
                greedy_result['fn_edges'] = greedy_result['fn_edges'] / greedy_result['gt_edges']
                greedy_result['identity_switches'] = greedy_result['identity_switches'] / greedy_result['gt_edges']
                greedy_result['fp_divisions'] = greedy_result['fp_divisions'] / greedy_result['gt_edges']
                greedy_result['fn_divisions'] = greedy_result['fn_divisions'] / greedy_result['gt_edges']
                greedy_result['sum'] = greedy_result['sum'] / greedy_result['gt_edges']
            writer.writerow(greedy_result)
            write_segments(segments_file % 'greedy', greedy_result['correct_segments'])
            tgmm_dbs = [
                    'linajea_160328_tgmm_SPM00_CM01_%s' % region,
                    'linajea_160328_tgmm_SPM01_CM00_%s' % region,
                    ]
            tgmm_dicts = [lev.get_tgmm_results(db_name, db_host).to_dict()
                          for db_name in tgmm_dbs]
            best_tgmm = None
            for tgmm_dict in tgmm_dicts:
                run_dict = {}
                for key in tgmm_dict.keys():
                    run_dict[key] = tgmm_dict[key][0]
                run_dict['region'] = region
                run_dict['model'] = "TGMM"
                run_dict['sum'] = (
                        run_dict['fn_edges'] +
                        run_dict['identity_switches'] +
                        run_dict['fn_divisions'] +
                        run_dict['fp_divisions'])
                if best_tgmm is None or best_tgmm['sum'] >\
                        run_dict['sum']:
                    best_tgmm = run_dict
            tgmm_dict = best_tgmm
            if args.normalize:
                tgmm_dict['fn_edges'] = tgmm_dict['fn_edges'] / tgmm_dict['gt_edges']
                tgmm_dict['identity_switches'] = tgmm_dict['identity_switches'] / tgmm_dict['gt_edges']
                tgmm_dict['fp_divisions'] = tgmm_dict['fp_divisions'] / tgmm_dict['gt_edges']
                tgmm_dict['fn_divisions'] = tgmm_dict['fn_divisions'] / tgmm_dict['gt_edges']
                tgmm_dict['sum'] = tgmm_dict['sum'] / tgmm_dict['gt_edges']
            writer.writerow(tgmm_dict)
            write_segments(segments_file % 'tgmm', tgmm_dict['correct_segments'])
