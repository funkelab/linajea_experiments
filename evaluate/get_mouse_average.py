import linajea.evaluation as lev
import csv
# from graph_from_csv import graph_from_csv
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_segments(outfile, results_list):
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
    setup = 'setup11_simple'
    eval_regions = ['early', 'middle', 'late']
    frame_sets = [[50, 100], [225, 275], [400, 450]]
    train_regions = ['early', 'middle', 'late']
    vald_regions = ['early', 'middle', 'late']
    sample = '140521'
    filter_params = {'version': 'v1.3-dev'}
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin"
    our_results = []
    greedy_results = []
    tgmm_results = []
    out_file = "mouse_average.csv"
    segments_file = 'mouse_average_segments.csv'

    # need tgmm, our best result on early, middle, late
    for region, frames in zip(eval_regions, frame_sets):
        segments_file = "mouse_segments_" + region + "_%s.csv"
        # side, model, fn, is, fp-d, fn-d, sum
        for train_region in train_regions:
            if train_region == region:
                continue
            vald_region = None
            vald_frames = None
            for vr, vf in zip(vald_regions, frame_sets):
                if vr != train_region and vr != region:
                    vald_region = vr
                    vald_frames = vf
                    break

            if vald_region is None:
                logger.error("ERROR: Vald region not found for eval region"
                             "%s and train region %s"
                             % (region, train_region))
            setup_with_train = setup + '_' + train_region
            logger.info("Getting best result for vald region %s and eval region %s",
                        vald_region, region)
            db_name = 'linajea_140521_setup11_simple_%s_%s_400000_te'
            vald_params = lev.get_best_result(
                    db_name % (train_region, vald_region),
                    db_host,
                    sample=sample,
                    frames=vald_frames,
                    filter_params=filter_params)
            logger.debug("Validation parameters: %s" % vald_params)

            eval_result = lev.get_result(
                    db_name % (train_region, region),
                    vald_params,
                    db_host,
                    sample=sample,
                    frames=frames)
            eval_result['model'] = 'ours'
            eval_result['sum'] = (
                    eval_result['fn_edges'] +
                    eval_result['identity_switches'] +
                    eval_result['fp_divisions'] +
                    eval_result['fn_divisions'])
            logger.debug("Evaluation result: %s" % eval_result)
            our_results.append(eval_result)
            # Get greedy tracking result for this region
            greedy_result = lev.get_greedy(
                    db_name % (train_region, region),
                    db_host,
                    frames=frames)
            greedy_result['model'] = 'greedy'
            greedy_result['sum'] = (
                    greedy_result['fn_edges'] +
                    greedy_result['identity_switches'] +
                    greedy_result['fp_divisions'] +
                    greedy_result['fn_divisions'])
            logger.debug("Greedy result: %s" % greedy_result)
            greedy_results.append(greedy_result)
            # Get tgmm score for this region
            tgmm_db = 'linajea_%s_tgmm' % sample
            tgmm_df = lev.get_tgmm_results(tgmm_db, db_host, frames=frames)
            tgmm_dict = tgmm_df.iloc[0].to_dict()
            print(tgmm_dict.keys())
            tgmm_dict['model'] = "TGMM"
            tgmm_dict['sum'] = (
                    tgmm_dict['fn_edges'] +
                    tgmm_dict['identity_switches'] +
                    tgmm_dict['fp_divisions'] +
                    tgmm_dict['fn_divisions'])
            tgmm_results.append(tgmm_dict)
    #for result in our_results:
    #    result['fn_edges'] = result['fn_edges'] / result['gt_edges']
    #    result['identity_switches'] = result['identity_switches'] / result['gt_edges']
    #    result['fp_divisions'] = result['fp_divisions'] / result['gt_edges']
    #    result['fn_divisions'] = result['fn_divisions'] / result['gt_edges']
    #    result['sum'] = result['sum'] / result['gt_edges']
    #for result in greedy_results:
    #    result['fn_edges'] = result['fn_edges'] / result['gt_edges']
    #    result['identity_switches'] = result['identity_switches'] / result['gt_edges']
    #    result['fp_divisions'] = result['fp_divisions'] / result['gt_edges']
    #    result['fn_divisions'] = result['fn_divisions'] / result['gt_edges']
    #    result['sum'] = result['sum'] / result['gt_edges']
    #for result in tgmm_results:
    #    result['fn_edges'] = result['fn_edges'] / result['gt_edges']
    #    result['identity_switches'] = result['identity_switches'] / result['gt_edges']
    #    result['fp_divisions'] = result['fp_divisions'] / result['gt_edges']
    #    result['fn_divisions'] = result['fn_divisions'] / result['gt_edges']
    #    result['sum'] = result['sum'] / result['gt_edges']

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
                    'identity_switches', 'fp_divisions', 'fn_divisions', 'sum'],
                extrasaction='ignore')
        writer.writeheader()
        writer.writerow(get_average(our_results, 'ours'))
        write_segments('mouse_segments_averaged_ours.csv', our_results)
        writer.writerow(get_average(greedy_results, 'greedy'))
        write_segments('mouse_segments_averaged_greedy.csv', greedy_results)
        writer.writerow(get_average(tgmm_results, 'TGMM'))
        write_segments('mouse_segments_averaged_tgmm.csv', tgmm_results)

        #graph_from_csv(out_file)
