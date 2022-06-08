import argparse
from copy import deepcopy
import csv
import logging
import os
import sys

import pandas as pd
import linajea.evaluation

from linajea import getNextInferenceData

logger = logging.getLogger(__name__)

def prune_report(report):
    score = {}
    if isinstance(report, dict):
        score.update(report)
    else:
        score.update(report.__dict__)
    for k in list(score.keys()):
        if "nodes" in k:
            del score[k]
    try:
        del score['fn_edge_list']
    except:
        pass
    return score


def write_segments(outfile, segments):
    with open(outfile, 'w') as f:
        seg_writer = csv.DictWriter(
                f,
                fieldnames=[
                    'time', 'correct', 'total', 'percent'],
                extrasaction='ignore')
        seg_writer.writeheader()
        print(list(segments.keys()))
        max_seg_length = max([int(k) for k in segments.keys()])
        for i in range(1, max_seg_length + 1):
            correct, total = segments[str(i)]
            if total == 0:
                assert correct == 0
            percent = correct / max(1, total)
            row = {'time': i, 'correct': correct,
                   'total': total, 'percent': percent}
            seg_writer.writerow(row)


def main():
    logging.basicConfig(
        level=20,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ])
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="config file")
    parser.add_argument('--db_name', help="db_name")
    parser.add_argument('--db_host', help="db_host")
    parser.add_argument('--validation', action="store_true",
                        help='use validation data?')
    parser.add_argument('--val', action="store_true",
                        help='get only val entries?')
    parser.add_argument('--validate_on_train', action="store_true",
                        help='validate on train data?')
    parser.add_argument('-p', '--param_id', type=int,
                        help='specfic result', default=None)
    parser.add_argument('--val_param_id', type=int,
                        help='specfic result', default=None)
    parser.add_argument('--checkpoint', type=int,
                        help='specific checkpoint/iteration (overwrites config)',
                        default=-1)
    parser.add_argument('-th', '--threshold', type=float,
                        help='cell score threshold (overwrites config)', default=None)
    parser.add_argument('-b', '--best', action="store_true",
                        help='best result')
    parser.add_argument('--sum', action="store_true",
                        help='sum samples')
    parser.add_argument('--sort', default="sum_errors",
                        help='sort by this column')
    parser.add_argument('--tag', type=str)
    args = parser.parse_args()

    score_columns = ['fp_edges', 'fn_edges', 'identity_switches',
                     'fp_divisions', 'fn_divisions']
    tmp_columns = [
         "cell_cycle_key",
         # "prefix",
         "matching_threshold",
         "filter_polar_bodies_key",
         "filter_polar_bodies",
         "weight_node_score",
         "selection_constant",
         "track_cost",
         "weight_division",
         "division_constant",
         "weight_child",
         "weight_continuation",
         "weight_edge_score",
         "rec_tracks",
         "rec_matched_tracks",
         "gt_edges",
         "rec_edges",
         "matched_edges",
         "gt_divisions",
         "rec_divisions",
         "iso_fp_division",
         "iso_fn_division",
         "fp_edges",
         "fn_edges",
         "identity_switches",
         "fp_divisions",
         "fn_divisions",
         #"num_error_free_tracks",
         #"num_rec_cells_last_frame",
         #"num_gt_cells_last_frame",
         "sum_divs",
         "sum_errors",
         "ratio_error_free_tracks",
         "node_recall",
         "edge_recall",
         "node_precision",
         "edge_precision"
    ]
    id_columns=[
         "cell_cycle_key",
         # "prefix",
         "filter_polar_bodies_key",
         "matching_threshold",
         "weight_node_score",
         "selection_constant",
         "track_cost",
         "weight_division",
         "division_constant",
         "weight_child",
         "weight_continuation",
         "weight_edge_score",
         ]
    val_columns=[
        "rec_tracks",
         "rec_matched_tracks",
         "gt_edges",
         "rec_edges",
         "matched_edges",
         "gt_divisions",
         "rec_divisions",
         "iso_fp_division",
         "iso_fn_division",
         "fp_edges",
         "fn_edges",
         "identity_switches",
         "fp_divisions",
         "fn_divisions",
         #"num_error_free_tracks",
         #"num_rec_cells_last_frame",
         #"num_gt_cells_last_frame",
         "sum_divs",
         "sum_errors"
         "ratio_error_free_tracks"
         "node_recall",
         "edge_recall",
         "node_precision",
         "edge_precision"
         ]
    tmp_header=[
         "cck",
         # "prefix",
         "m_th",
         "pbk",
         "pb",
         "w_n_s",
         "s_c",
         "t_c",
         "w_d",
         "d_c",
         "w_ch",
         "w_co",
         "w_e_sc",
         "rec_tr",
         "r_ma_tr",
         "gt_e",
         "rec_e",
         "ma_e",
         "gt_div",
         "rec_div",
         "i_fp_div",
         "i_fn_div",
         "fp_e",
         "fn_e",
         "id_sw",
         "fp_div",
         "fn_div",
         #"nef_lf",
         #"nrc_lf",
         #"ngtc_lf",
         "sum_divs",
         "sum_errors",
         "r_e_fr_tr",
         "nr",
         "er",
         "np",
         "ep"
        ]

    results = {}
    results_sum = {}
    sample = None
    results_df = []
    if args.db_name is not None:
        res = linajea.evaluation.get_results_sorted_db(
            args.db_name, args.db_host,
            score_columns=score_columns)
        # print(prune_report(res))
        try:
            res['prefix'] = res['prefix'].map(lambda a: os.path.dirname(str(a)))
        except:
            pass

        try:
            res['cell_cycle_key'] = res['cell_cycle_key'].map(lambda a: "T"
                                                              if (isinstance(a, str) and a != "") else "F")
        except:
            pass

        try:
            res['filter_polar_bodies_key'] = res['filter_polar_bodies_key'].map(
                lambda a: "T" if (isinstance(a, str) and a != "") else "F")
        except:
            pass


        for col in tmp_columns:
            if col not in res:
                res[col] = None

        columns = []
        header = []
        for c,h in zip(tmp_columns, tmp_header):
            if c in res.columns:
                columns.append(c)
                header.append(h)
        res.to_csv("test.csv",
                   float_format="%.3f",
                   columns=columns, header=header)
        print(res.to_string(float_format="%.5f",
                            columns=columns, header=header))
    else:
        for inf_config in getNextInferenceData(args, is_evaluate=True):
            inf_config.evaluate.parameters.filter_polar_bodies = False
            if args.threshold is not None:
                if inf_config.inference.cell_score_threshold != args.threshold:
                    continue

            tmp_sample = inf_config.inference.data_source.datafile.filename
            if tmp_sample == sample:
                continue
            logger.info("data: {}".format(inf_config.inference))
            sample = tmp_sample

            print(sample)
            if args.param_id is not None:
                # inf_config.evaluate.parameters = {
                #     'roi': {'offset': [0, 0, 0, 0],
                #             'shape': [270] + inf_config.inference.data_source.datafile.file_roi.shape[1:]}
                # }
                res = linajea.evaluation.get_result_id(
                    inf_config,
                    args.param_id)
                print(res)
                segments_file = f"celegans_segments__{os.path.basename(inf_config.general.setup_dir)}__{os.path.basename(sample)}__{args.tag}__{args.param_id}.csv"
                write_segments(segments_file, res['correct_segments'])
                exit()
            elif args.best:
                res = linajea.evaluation.get_best_result_with_config(
                    inf_config,
                    score_columns=score_columns)
                print(prune_report(res))
            else:
                res = linajea.evaluation.get_results_sorted(
                    inf_config,
                    filter_params={"val": args.val},
                    score_columns=score_columns,
                    sort_by=args.sort)
                results[os.path.basename(sample)] = res

                try:
                    res['prefix'] = res['prefix'].map(lambda a: os.path.dirname(str(a)))
                except:
                    pass

                try:
                    res['cell_cycle_key'] = res['cell_cycle_key'].map(
                        lambda a: "T"
                        if (isinstance(a, str) and a != "") else "F")
                except:
                    pass

                try:
                    res['filter_polar_bodies_key'] = res['filter_polar_bodies_key'].map(
                        lambda a: "T" if (isinstance(a, str) and a != "") else "F")
                except:
                    pass

                try:
                    res['ratio_error_free_tracks'] = (
                        res["num_error_free_tracks"]/res["num_gt_cells_last_frame"])
                except:
                    pass

                for col in tmp_columns:
                    if col not in res:
                        res[col] = None

                results_sum[os.path.basename(sample)] = res.reset_index()

                results_df.append(res)
                if not args.sum:
                    columns = []
                    header = []
                    for c,h in zip(tmp_columns, tmp_header):
                        if c in res.columns:
                            columns.append(c)
                            header.append(h)
                    res.to_csv("test.csv",
                               float_format="%.3f",
                               columns=columns, header=header)
                    print(res.to_string(float_format="%.5f",
                                        columns=columns, header=header))


    results_df = pd.concat(results_df).groupby([
        'cell_cycle_key', 'weight_node_score', 'selection_constant',
        'track_cost', 'weight_division', 'division_constant', 'weight_child',
        'weight_continuation', 'weight_edge_score']).sum().reset_index()
    results_df.sort_values('sum_errors', inplace=True)
    results_df = results_df[results_df['matching_threshold'] >=
                            results_df['matching_threshold'].mean()]
    print(results_df.to_string(float_format="%.3f",
                               columns=columns, header=header))
    if args.sum:
        t = pd.concat(list(results_sum.values())).reset_index()
        del t['param_id']
        del t['_id']

        res_df_count = t.groupby(id_columns, dropna=False, as_index=False).count()
        max_count = res_df_count['sum_errors'].max()
        print("max num datasets per exp", max_count)
        res_df = t.groupby(id_columns, dropna=False, as_index=False).agg(
            lambda x: -1 if len(x) != max_count else sum(x))

        res_df = res_df[res_df.sum_errors != -1]
        sort_by = 'sum_errors'
        ascending = True
        res_df.sort_values(sort_by, ascending=ascending, inplace=True)
        columns = []
        header = []
        for c,h in zip(tmp_columns, tmp_header):
            if c in res_df.columns:
                columns.append(c)
                header.append(h)
        print(res_df.to_string(float_format="%.3f",
                               columns=columns, header=header))


if __name__ == "__main__":
    main()
