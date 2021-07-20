import linajea.evaluation as lev
import csv


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

    with open(outfile, 'w') as f:
        seg_writer = csv.DictWriter(
                f,
                fieldnames=[
                    'time', 'percent'],
                extrasaction='ignore')
        seg_writer.writeheader()
        for i in range(1, max_seg_length + 1):
            average = sum(p[str(i)] for p in percents_list) / len(percents_list)
            row = {'time': i, 'percent': average}
            seg_writer.writerow(row)


if __name__ == '__main__':
    setup = 'setup111_simple'
    regions = ['eval_side_1', 'eval_side_2']
    sample = '160328'
    db_host = "localhost"  # TODO: Replace with MongoDB URL
    # need tgmm, our best result on side 1 and side 2
    # model, gt_edges, fn, is, fp-d, fn-d, sum
    fieldnames = ['model', 'gt_edges', 'fn_edges',
            'identity_switches', 'fp_divisions', 'fn_divisions', 'sum']
    our_results = []
    greedy_results = []
    tgmm_results = []
    out_file = "zebrafish_average.csv"
    for region in regions:
            print("getting best result for region %s" % region)
            db_name = "linajea_160328_setup111_simple_%s_400000_te" % region
            best_dict = lev.get_best_result(db_name, db_host)
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
            greedy_result['fn_edges'] = greedy_result['fn_edges'] / greedy_result['gt_edges']
            greedy_result['identity_switches'] = greedy_result['identity_switches'] / greedy_result['gt_edges']
            greedy_result['fp_divisions'] = greedy_result['fp_divisions'] / greedy_result['gt_edges']
            greedy_result['fn_divisions'] = greedy_result['fn_divisions'] / greedy_result['gt_edges']
            greedy_result['sum'] = greedy_result['sum'] / greedy_result['gt_edges']
            greedy_results.append(greedy_result)

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
        write_segments('zebra_segments_averaged_ours.csv', our_results)
        writer.writerow(get_average(greedy_results, 'greedy'))
        write_segments('zebra_segments_averaged_greedy.csv', greedy_results)
        writer.writerow(get_average(tgmm_results, 'TGMM'))
        write_segments('zebra_segments_averaged_tgmm.csv', tgmm_results)
