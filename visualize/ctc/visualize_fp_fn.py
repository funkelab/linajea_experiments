import argparse
import logging
import os
import sys

import networkx as nx

import daisy
import linajea
import linajea.tracking
import linajea.evaluation
from linajea.visualization.ctc import write_ctc


logging.basicConfig(level=logging.INFO)


def export_db(
        candidate_db_name,
        selected_key,
        gt_db_name,
        start_frame,
        end_frame,
        sample,
        output_dir):

    assert end_frame > start_frame

    txt_fn = "man_track.txt"
    tif_fn = "man_track{:03d}.tif"
    data_config = linajea.load_config(
        os.path.join(sample, "data_config.toml"))
    shape = data_config['general']['shape']

    voxel_size = data_config['general']['resolution']
    output_dir = os.path.join(output_dir, output_dir)

    def create_graph_and_write_ctc(src_graph, edges, name, gt, nodes=None):
        trgt_graph = nx.DiGraph()
        if edges is not None:
            for edge in edges:
                trgt_graph.add_node(edge[0], **src_graph.nodes(data=True)[edge[0]])
                trgt_graph.add_node(edge[1], **src_graph.nodes(data=True)[edge[1]])
                trgt_graph.add_edge(edge[0], edge[1])
        if nodes is not None:
            for node in nodes:
                trgt_graph.add_node(node, **src_graph.nodes(data=True)[node])

        trgt_graph = linajea.tracking.TrackGraph(trgt_graph, frame_key='t')

        write_ctc(trgt_graph, start_frame, end_frame, shape,
                  output_dir + name, txt_fn, tif_fn,
                  voxel_size=voxel_size, gt=gt)


    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 1e10, 1e10, 1e10))
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea"
    db = linajea.CandidateDatabase(
            candidate_db_name, db_host, parameters_id=selected_key)
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)

    print("Reading GT cells and edges in %s" % roi)
    gt_subgraph = gt_db[roi]
    gt_graph = linajea.tracking.TrackGraph(gt_subgraph, frame_key='t', roi=roi)
    gt_tracks = list(gt_graph.get_tracks())
    print("Found %d GT tracks" % len(gt_tracks))

    print("Reading cells and edges in %s" % roi)
    rec_subgraph = db.get_selected_graph(roi)
    rec_graph = linajea.tracking.TrackGraph(rec_subgraph, frame_key='t', roi=roi)
    rec_tracks = list(rec_graph.get_tracks())
    print("Found %d tracks" % len(rec_tracks))

    m = linajea.evaluation.match_edges(
        gt_graph, rec_graph,
        matching_threshold=15)
    (gt_edges, rec_edges, edge_matches, unselected_potential_matches) = m
    edge_matches = [(gt_edges[gt_ind], rec_edges[rec_ind])
                    for gt_ind, rec_ind in edge_matches]

    validation_score = False
    ignore_one_off_div_errors = True
    fn_div_count_unconnected_parent = False
    window_size = 270
    sparse = False
    evaluator = linajea.evaluation.Evaluator(
            gt_graph,
            rec_graph,
            edge_matches,
            unselected_potential_matches,
            sparse=sparse,
            validation_score=validation_score,
            window_size=window_size,
            ignore_one_off_div_errors=ignore_one_off_div_errors,
            fn_div_count_unconnected_parent=fn_div_count_unconnected_parent)
    report = evaluator.evaluate()

    matched_rec_edges = set([match[1] for match in edge_matches])
    fp_edges = report.fp_edge_list

    matched_gt_edges = set([match[0] for match in edge_matches])
    fn_edges = report.fn_edge_list


    create_graph_and_write_ctc(gt_graph, matched_gt_edges, "gt_tp", True)
    create_graph_and_write_ctc(gt_graph, fn_edges, "gt_fn", True)
    create_graph_and_write_ctc(gt_graph, None, "gt_is", True, nodes=report.identity_switch_gt_nodes)
    fn_div_nodes = (report.no_connection_gt_nodes +
                    report.unconnected_child_gt_nodes +
                    report.unconnected_parent_gt_nodes)
    create_graph_and_write_ctc(gt_graph, None, "gt_fn_div", True, nodes=fn_div_nodes)
    create_graph_and_write_ctc(rec_graph, matched_rec_edges, "rec_tp", False)
    create_graph_and_write_ctc(rec_graph, fp_edges, "rec_fp", False)
    create_graph_and_write_ctc(rec_graph, None, "rec_fp_div", False, nodes=report.fp_div_rec_nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cand_db_name', required=True,
                        help='name of candidate database')
    parser.add_argument('--gt_db_name', required=True,
                        help='name of gt database')
    parser.add_argument('--param_id', required=True, type=int,
                        help='id of used parameter set (in candidate db)')
    parser.add_argument('--start_frame', required=True, type=int)
    parser.add_argument('--end_frame', required=True, type=int)
    parser.add_argument('--sample', required=True,
                        help='sample file to be used')
    parser.add_argument('--output_dir', required=True,
                        help='output directory')

    args = parser.parse_args()

    export_db(
        args.cand_db_name,
        args.param_id,
        args.gt_db_name,
        args.start_frame,
        args.end_frame,
        args.sample,
        args.output_dir)
