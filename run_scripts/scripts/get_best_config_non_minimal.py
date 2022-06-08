import argparse
from copy import deepcopy
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
    parser.add_argument('-p', '--param_id', type=int,
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
    args = parser.parse_args()

    score_columns = ['fp_edges', 'fn_edges', 'identity_switches',
                     'fp_divisions', 'fn_divisions']
    tmp_columns = [
         "use_cell_state",
         "prefix",
         "matching_threshold",
         "cost_appear",
         "cost_disappear",
         "cost_split",
         "cost_daughter",
         "cost_normal",
         "clip_low_score",
         "threshold_node_score",
         "weight_node_score",
         "threshold_edge_score",
         "threshold_split_score",
         "threshold_is_normal_score",
         "weight_prediction_distance_cost",
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
         "sum_divs",
         "sum_errors"
         ]
    id_columns=[
         "use_cell_state",
         "prefix",
         "matching_threshold",
         "cost_appear",
         "cost_disappear",
         "cost_split",
         "cost_daughter",
         "cost_normal",
         "clip_low_score",
         "threshold_node_score",
         "weight_node_score",
         "threshold_edge_score",
         "threshold_split_score",
         "threshold_is_normal_score",
         "weight_prediction_distance_cost",
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
         "sum_divs",
         "sum_errors"
         ]
    tmp_header=[
         "ucs",
         "prefix",
         "m_th",
         "c_a",
         "c_d",
         "c_s",
         "c_da",
         "c_n",
         "c_sc",
         "th_n_sc",
         "w_n_sc",
         "th_e_sc",
         "th_ss",
         "th_isn",
         "w_p_d_c",
         "rec_tr",
         "r_ma_tr",
         "gt_e",
         "rec_e",
         "ma_e",
         "gt_d",
         "rec_d",
         "ifpd",
         "ifnd",
         "fp_e",
         "fn_e",
         "idsw",
         "fp_d",
         "fn_d",
         "sum_d",
         "sum_e"
        ]

    results = {}
    results_sum = {}
    sample = None
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
            res['cell_cycle_key'] = res['cell_cycle_key'].map(
                lambda a: isinstance(a, str) and a != "")
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
        print(res.to_string(float_format="%.3f",
                            columns=columns, header=header))
    else:
        for inf_config in getNextInferenceData(args, is_evaluate=True):
            if args.threshold is not None:
                print(inf_config.inference.cell_score_threshold, args.threshold)
                if inf_config.inference.cell_score_threshold != args.threshold:
                    continue

            tmp_sample = inf_config.inference.data_source.datafile.filename
            print(tmp_sample)
            if tmp_sample == sample:

                continue
            sample = tmp_sample

            print(sample)
            if args.param_id is not None:
                res = linajea.evaluation.get_result_id(
                    inf_config,
                    args.param_id)
                results[os.path.basename(sample)] = prune_report(res)
            else:
                if args.best:
                    res = linajea.evaluation.get_best_result_with_config(
                        inf_config,
                        score_columns=score_columns)
                else:
                    res = linajea.evaluation.get_results_sorted(
                        inf_config,
                        score_columns=score_columns,
                        sort_by=args.sort)
                results[os.path.basename(sample)] = res

                try:
                    res['prefix'] = res['prefix'].map(lambda a: os.path.dirname(str(a)))
                except:
                    pass

                for col in tmp_columns:
                    if col not in res:
                        res[col] = None

                results_sum[os.path.basename(sample)] = res.reset_index()

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
                    print(res.to_string(float_format="%.3f",
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
