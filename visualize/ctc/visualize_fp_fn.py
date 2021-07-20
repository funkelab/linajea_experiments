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
    roi = daisy.Roi((start_frame, 0, 0, 0),
                    (end_frame - start_frame, 1e10, 1e10, 1e10))
    db_host = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea"
    db = linajea.CandidateDatabase(
            candidate_db_name, db_host, parameters_id=selected_key)
    gt_db = linajea.CandidateDatabase(gt_db_name, db_host)

    print("Reading GT cells and edges in %s" % roi)
    gt_subgraph = gt_db[roi]
    gt_graph = linajea.tracking.TrackGraph(gt_subgraph, frame_key='t')
    gt_tracks = list(gt_graph.get_tracks())
    print("Found %d GT tracks" % len(gt_tracks))

    print("Reading cells and edges in %s" % roi)
    rec_subgraph = db.get_selected_graph(roi)
    rec_graph = linajea.tracking.TrackGraph(rec_subgraph, frame_key='t')
    rec_tracks = list(rec_graph.get_tracks())
    print("Found %d tracks" % len(rec_tracks))

    m = linajea.evaluation.match_edges(
        gt_graph, rec_graph,
        matching_threshold=15)
    (edges_x, edges_y, edge_matches, edge_fps) = m

    matched_rec_edges_id = set([match[1] for match in edge_matches])
    matched_rec_edges = set()
    for edge_index in matched_rec_edges_id:
        edge = edges_y[edge_index]
        matched_rec_edges.add(edge)
    rec_edges = set(rec_graph.edges)
    fp_edges = list(rec_edges - matched_rec_edges)
    rec_track_tp = nx.DiGraph()
    rec_track_fp = nx.DiGraph()
    for edge in matched_rec_edges:
        rec_track_tp.add_node(edge[0], **rec_graph.nodes(data=True)[edge[0]])
        rec_track_tp.add_node(edge[1], **rec_graph.nodes(data=True)[edge[1]])
        rec_track_tp.add_edge(edge[0], edge[1])
    for edge in fp_edges:
        rec_track_fp.add_node(edge[0], **rec_graph.nodes(data=True)[edge[0]])
        rec_track_fp.add_node(edge[1], **rec_graph.nodes(data=True)[edge[1]])
        rec_track_fp.add_edge(edge[0], edge[1])

    rec_track_tp = linajea.tracking.TrackGraph(rec_track_tp, frame_key='t')
    rec_track_fp = linajea.tracking.TrackGraph(rec_track_fp, frame_key='t')

    matched_gt_edges_id = set([match[0] for match in edge_matches])
    matched_gt_edges = set()
    for edge_index in matched_gt_edges_id:
        edge = edges_x[edge_index]
        matched_gt_edges.add(edge)
    gt_edges = set(gt_graph.edges)
    fn_edges = list(gt_edges - matched_gt_edges)
    gt_track_tp = nx.DiGraph()
    gt_track_fn = nx.DiGraph()
    for edge in matched_gt_edges:
        gt_track_tp.add_node(edge[0], **gt_graph.nodes(data=True)[edge[0]])
        gt_track_tp.add_node(edge[1], **gt_graph.nodes(data=True)[edge[1]])
        gt_track_tp.add_edge(edge[0], edge[1])
    for edge in fn_edges:
        gt_track_fn.add_node(edge[0], **gt_graph.nodes(data=True)[edge[0]])
        gt_track_fn.add_node(edge[1], **gt_graph.nodes(data=True)[edge[1]])
        gt_track_fn.add_edge(edge[0], edge[1])

    gt_track_tp = linajea.tracking.TrackGraph(gt_track_tp, frame_key='t')
    gt_track_fn = linajea.tracking.TrackGraph(gt_track_fn, frame_key='t')

    txt_fn = "man_track.txt"
    tif_fn = "man_track{:03d}.tif"
    data_config = linajea.load_config(
        os.path.join(sample, "data_config.toml"))
    shape = data_config['general']['shape']

    # voxel_size = data_config['general']['resolution']
    voxel_size = [1, 5, 1, 1]
    write_ctc(gt_track_tp, start_frame, end_frame, shape,
              output_dir + "gt_tp", txt_fn, tif_fn,
              voxel_size=voxel_size, gt=True)
    write_ctc(gt_track_fn, start_frame, end_frame, shape,
              output_dir + "gt_fn", txt_fn, tif_fn,
              voxel_size=voxel_size, gt=True)
    write_ctc(rec_track_tp, start_frame, end_frame, shape,
              output_dir + "rec_tp", txt_fn, tif_fn,
              voxel_size=voxel_size)
    write_ctc(rec_track_fp, start_frame, end_frame, shape,
              output_dir + "rec_fp", txt_fn, tif_fn,
              voxel_size=voxel_size)


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
