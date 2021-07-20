import linajea.evaluation as lev
import csv
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


def get_split_name(train_region, eval_region):
    if eval_region == "early":
        if train_region == "late":
            return "early_1"
        elif train_region == "middle":
            return "early_2"
    if eval_region == "middle":
        if train_region == "late":
            return "middle_1"
        if train_region == "early":
            return "middle_2"
    if eval_region == "late":
        if train_region == "middle":
            return "late_1"
        if train_region == "early":
            return "late_2"
    raise ValueError("No split name for train region %s and eval region %s"
                     % (train_region, eval_region))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--normalize', action='store_true')
    args = parser.parse_args()
    setup = 'setup11_simple'
    eval_regions = ['early', 'middle', 'late']
    frame_sets = [[50, 100], [225, 275], [400, 450]]
    train_regions = ['early', 'middle', 'late']
    vald_regions = ['early', 'middle', 'late']
    sample = '140521'
    filter_params = {'version': 'v1.3-dev'}
    db_host = "localhost"

    # need tgmm, our best result on early, middle, late
    for region, frames in zip(eval_regions, frame_sets):
        # Get tgmm score for this region
        tgmm_db = 'linajea_140521_tgmm'
        tgmm_df = lev.get_tgmm_results(tgmm_db, db_host, frames=frames)
        tgmm_dict = tgmm_df.iloc[0].to_dict()
        print(tgmm_dict.keys())
        tgmm_dict['model'] = "TGMM"
        tgmm_dict['sum'] = (
                tgmm_dict['fn_edges'] +
                tgmm_dict['identity_switches'] +
                tgmm_dict['fp_divisions'] +
                tgmm_dict['fn_divisions'])
        if args.normalize:
            tgmm_dict['fn_edges'] = tgmm_dict['fn_edges'] / tgmm_dict['gt_edges']
            tgmm_dict['identity_switches'] = tgmm_dict['identity_switches'] / tgmm_dict['gt_edges']
            tgmm_dict['fp_divisions'] = tgmm_dict['fp_divisions'] / tgmm_dict['gt_edges']
            tgmm_dict['fn_divisions'] = tgmm_dict['fn_divisions'] / tgmm_dict['gt_edges']
            tgmm_dict['sum'] = tgmm_dict['sum'] / tgmm_dict['gt_edges']
        for train_region in train_regions:
            if train_region == region:
                continue
            split_name = get_split_name(train_region, region)
            out_file = "mouse_" + split_name + ".csv"
            segments_file = "mouse_segments_" + split_name + "_%s.csv"

            to_write = {}
            # side, model, fn, is, fp-d, fn-d, sum
            with open(out_file, 'w') as f:
                writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            'model', 'gt_edges', 'fn_edges',
                            'identity_switches', 'fp_divisions', 'fn_divisions', 'sum'],
                        extrasaction='ignore')
                writer.writeheader()
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
                if args.normalize:
                    eval_result['fn_edges'] = eval_result['fn_edges'] / eval_result['gt_edges']
                    eval_result['identity_switches'] = eval_result['identity_switches'] / eval_result['gt_edges']
                    eval_result['fp_divisions'] = eval_result['fp_divisions'] / eval_result['gt_edges']
                    eval_result['fn_divisions'] = eval_result['fn_divisions'] / eval_result['gt_edges']
                    eval_result['sum'] = eval_result['sum'] / eval_result['gt_edges']
                    greedy_result['fn_edges'] = greedy_result['fn_edges'] / greedy_result['gt_edges']
                    greedy_result['identity_switches'] = greedy_result['identity_switches'] / greedy_result['gt_edges']
                    greedy_result['fp_divisions'] = greedy_result['fp_divisions'] / greedy_result['gt_edges']
                    greedy_result['fn_divisions'] = greedy_result['fn_divisions'] / greedy_result['gt_edges']
                    greedy_result['sum'] = greedy_result['sum'] / greedy_result['gt_edges']
                writer.writerow(eval_result)
                writer.writerow(greedy_result)
                write_segments(segments_file % 'ours', eval_result['correct_segments'])
                write_segments(segments_file % 'greedy', greedy_result['correct_segments'])

                writer.writerow(tgmm_dict)
                write_segments(segments_file % 'tgmm', tgmm_dict['correct_segments'])
