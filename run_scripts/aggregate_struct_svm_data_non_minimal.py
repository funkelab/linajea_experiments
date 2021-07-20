import argparse
import logging
import time
import os
import sys
import glob

import networkx as nx
import numpy as np

from linajea import CandidateDatabase
import daisy

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

class Indicator(object):
    def __init__(self):
        self.features = {}
        self.features["node_appear"] = 0
        self.features["node_disappear"] = 0
        self.features["node_continuation_weight"] = 0
        self.features["node_continuation_weighttimesthreshold"] = 0
        self.features["node_select_weight"] = 0
        self.features["node_select_weighttimesthreshold"] = 0
        self.features["node_daughter_weight"] = 0
        self.features["node_daughter_weighttimesthreshold"] = 0
        self.features["node_split_weight"] = 0
        self.features["node_split_weighttimesthreshold"] = 0
        self.features["edge_select_weight"] = 0
        self.features["edge_select_weighttimesthreshold"] = 0
        self.features["edge_split_weight"] = 0

class Node(object):
    def __init__(self):
        self.indicator = {}
        self.indicator["node_selected"] = None
        self.indicator["node_appear"] = None
        self.indicator["node_disappear"] = None
        self.indicator["node_split"] = None
        self.indicator["node_cycle"] = None
        self.indicator["node_is_daughter"] = None
        self.indicator["node_is_normal"] = None


class Edge(object):
    def __init__(self):
        self.indicator = {}
        self.indicator["edge_selected"] = None
        self.indicator["edge_split"] = None
        # self.indicator["edge_to_mother"] = None
        # self.indicator["edge_from_daughter"] = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='path to files')
    parser.add_argument('--db_name', type=str,
                        help='db_name')
    parser.add_argument('--gt_db_name', type=str,
                        help='gt db_name')
    parser.add_argument('--db_host', type=str,
                        default="mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea",
                        help='db host')
    parser.add_argument('--debug', type=int, default=None,
                        help='limit number of blocks written')
    args = parser.parse_args()

    logger.info("loading features...")
    feature_blocks = {}
    feature_files = list(glob.glob(os.path.join(args.dir, "features_*")))
    for fl in feature_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        feature_blocks.setdefault(block_id, {})

    for fl in feature_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        kind = fl.split("features_")[-1].rsplit("_", 1)[0]
        with open(fl, 'r') as f:
            for ln in f:
                idx, val = ln.split()
                val = float(val)
                feature_blocks[block_id].setdefault(int(idx), Indicator()).features[kind] = val
                # node = feature_blocks[block_id].setdefault(idx, Indicator())
                # node.features[kind] = val
                # print(node.features)
                # print(feature_blocks[block_id].setdefault(idx, Indicator()).features)
                # exit()
    print(sorted(feature_blocks.keys()))
    ind_cntr = 0
    with open("features.txt", 'w') as f:
        for bidx, block_id in enumerate(sorted(feature_blocks.keys())):
            if args.debug is not None and bidx >= args.debug:
                break

            block = feature_blocks[block_id]
            assert sorted(list(block.keys())) == list(range(len(block))),\
                "some error in reading node features"

            for nidx in sorted(block.keys()):
                if len(block[nidx].features) != 13:
                    print(block[nidx].features)
                # f.write(("{} "*(len(block[nidx].features))).format(*block[nidx].features.values()) + "\n")

                f.write(("{} "*(len(block[nidx].features))).format(
                    block[nidx].features["node_appear"],
                    block[nidx].features["node_disappear"],
                    block[nidx].features["node_continuation_weight"],
                    block[nidx].features["node_continuation_weighttimesthreshold"],
                    block[nidx].features["node_select_weight"],
                    block[nidx].features["node_select_weighttimesthreshold"],
                    block[nidx].features["node_daughter_weight"],
                    block[nidx].features["node_daughter_weighttimesthreshold"],
                    block[nidx].features["node_split_weight"],
                    block[nidx].features["node_split_weighttimesthreshold"],
                    block[nidx].features["edge_select_weight"],
                    block[nidx].features["edge_select_weighttimesthreshold"],
                    block[nidx].features["edge_split_weight"]) + "\n")
                # f.write(("{} "*(len(block[nidx].features) + 1)).format(nidx + ind_cntr, *block[nidx].features.values()) + "\n")

            ind_cntr += len(block.keys())


    logger.info("loading constraints...")
    constraint_blocks = {}
    constraint_files = list(glob.glob(os.path.join(args.dir, "constraints_*")))
    for fl in constraint_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        constraint_blocks.setdefault(block_id, [])

    for fl in constraint_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        kind = fl.split("constraints_")[-1].rsplit("_", 1)[0]
        with open(fl, 'r') as f:
            for ln in f:
                constraint_blocks[block_id].append(ln)

    ind_cntr = 0
    with open("constraints.txt", 'w') as f:
        for bidx, block_id in enumerate(sorted(constraint_blocks.keys())):
            if args.debug is not None and bidx >= args.debug:
                break

            block = constraint_blocks[block_id]
            feat_block = feature_blocks[block_id]
            for cnstr in block:
                cnstr = cnstr.split()
                for i in range(len(cnstr)):
                    if "*" in cnstr[i]:
                        fs = cnstr[i].split("*")
                        fs[1] = str(int(fs[1]) + ind_cntr)
                        cnstr[i] = '*'.join(fs)
                cnstr = " ".join(cnstr)
                f.write("{}\n".format(cnstr))
            ind_cntr += len(feat_block)

    print(sorted(constraint_blocks.keys()))

    logger.info("loading indicators...")
    logger.info("...nodes")
    roi = daisy.Roi(offset=[0, 0, 0, 0], shape=[9999, 9999, 9999, 9999])
    db_graph = CandidateDatabase(args.db_name, args.db_host, mode='r')[roi]
    gt_db_graph = CandidateDatabase(args.gt_db_name, args.db_host, mode='r')[roi]
    graph_blocks = {}
    # best_effort_blocks = {}

    node_blocks = {}
    node_files = list(glob.glob(os.path.join(args.dir, "node_*")))
    for fl in node_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        node_blocks.setdefault(block_id, {})
        graph_blocks.setdefault(block_id, nx.Graph())
        # best_effort_blocks.setdefault(block_id, {})

    for fl in node_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        kind = fl.rsplit("_", 1)[0]
        with open(fl, 'r') as f:
            for ln in f:
                idx, val = ln.split()
                val = int(val)
                idx = int(idx)
                node_blocks[block_id].setdefault(idx, Node()).indicator[kind] = val
                # print(db_graph.nodes()[idx])
                if "probable_gt_state" in db_graph.nodes()[idx]:
                    graph_blocks[block_id].add_node(idx)

    logger.info("...edges")
    edge_blocks = {}
    edge_files = list(glob.glob(os.path.join(args.dir, "edge_*")))
    for fl in edge_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        edge_blocks.setdefault(block_id, {})


    for fl in edge_files:
        fl = os.path.basename(fl)
        block_id = int(fl.split("_")[-1][1:])
        kind = fl.rsplit("_", 1)[0]
        with open(fl, 'r') as f:
            for ln in f:
                n1, n2, val = ln.split()
                val = int(val)
                n1 = int(n1)
                n2 = int(n2)
                if db_graph.nodes()[n2]['t'] < db_graph.nodes()[n1]['t']:
                    n1, n2 = n2, n1
                edge_blocks[block_id].setdefault((n1, n2), Edge()).indicator[kind] = val

                if "probable_gt_cell_id" in db_graph.nodes()[n1] and \
                   "probable_gt_cell_id" in db_graph.nodes()[n2]:
                    n1_prob_gt_cell_id = db_graph.nodes()[n1]["probable_gt_cell_id"]
                    n2_prob_gt_cell_id = db_graph.nodes()[n2]["probable_gt_cell_id"]
                    if (n1_prob_gt_cell_id, n2_prob_gt_cell_id) in gt_db_graph.edges() or \
                       (n2_prob_gt_cell_id, n1_prob_gt_cell_id) in gt_db_graph.edges():
                        # print("found gt edge {} {}".format(n1_prob_gt_cell_id, n2_prob_gt_cell_id))
                        graph_blocks[block_id].add_edge(n1, n2)


    print(sorted(graph_blocks.keys()))
    print(sorted(node_blocks.keys()))
    print(sorted(edge_blocks.keys()))
    logger.info("loading best effort...")
    ind_cntr = 0
    cnt_split = 0
    cnt_selected = 0
    cnt_appear = 0
    cnt_disappear = 0
    cnt_is_daughter = 0
    cnt_is_normal = 0
    cnt_edge_split = 0
    cnt_edge_selected = 0
    best_effort = {}
    for bidx, block_id in enumerate(sorted(graph_blocks.keys())):
        if args.debug is not None and bidx >= args.debug:
            break

        graph = graph_blocks[block_id]
        # best_effort = best_effort_blocks[block_id]
        nodes = node_blocks[block_id]
        edges = edge_blocks[block_id]
        feat_block = feature_blocks[block_id]

        for cc in sorted(nx.connected_components(graph)):
            subgraph = graph.subgraph(cc).copy()
            for n in subgraph.nodes():
                # all nodes in graph are selected (have a unique gt match)
                # print(nodes[n].indicator)
                idx = nodes[n].indicator["node_selected"] + ind_cntr
                best_effort[idx] = 1
                cnt_selected += 1
                d = graph.degree[n]
                # assert d > 0, "invalid node, unconnected, check"
                if d <= 0:
                    print("invalid node, unconnected, check {} {} {}".format(d, n, db_graph.nodes()[n]))
                    idx = nodes[n].indicator["node_appear"] + ind_cntr
                    best_effort[idx] = 1
                    cnt_appear += 1
                    idx = nodes[n].indicator["node_disappear"] + ind_cntr
                    best_effort[idx] = 1
                    cnt_disappear += 1
                if d == 3:
                    # if node has 3 neighbors -> split
                    idx = nodes[n].indicator["node_split"] + ind_cntr
                    assert idx is not None, "check2"
                    best_effort[idx] = 1
                    cnt_split += 1
                elif d == 1:
                    # if node has 1 neighbor -> appear or disappear
                    assert len(list(graph.neighbors(n))) == 1, "check neighbors"
                    neighb = [ne for ne in graph.neighbors(n)][0]
                    nt = db_graph.nodes()[n]['t']
                    neighbt = db_graph.nodes()[neighb]['t']
                    if nt < neighbt:
                        # if neighb after -> appear
                        idx = nodes[n].indicator["node_appear"] + ind_cntr
                        best_effort[idx] = 1
                        cnt_appear += 1
                    else:
                        # if neighb before -> disappear
                        idx = nodes[n].indicator["node_disappear"] + ind_cntr
                        best_effort[idx] = 1
                        cnt_disappear += 1
            for n1, n2 in subgraph.edges():
                if db_graph.nodes()[n2]['t'] < db_graph.nodes()[n1]['t']:
                    n1, n2 = n2, n1
                if (nodes[n1].indicator["node_split"] + ind_cntr) in best_effort:
                    assert (nodes[n2].indicator["node_split"] + ind_cntr) not in best_effort, 'check3'
                    # if predecessor split -> node is daughter
                    idx = nodes[n2].indicator["node_is_daughter"] + ind_cntr
                    if idx not in best_effort:
                        cnt_is_daughter += 1
                    best_effort[idx] = 1
                    idx = edges[(n1, n2)].indicator["edge_split"] + ind_cntr
                    if idx not in best_effort:
                        cnt_edge_split += 1
                    best_effort[idx] = 1
                elif (nodes[n2].indicator["node_split"] + ind_cntr) in best_effort:
                    idx = edges[(n1, n2)].indicator["edge_split"] + ind_cntr
                    if idx not in best_effort:
                        cnt_edge_split += 1
                    best_effort[idx] = 1
                    # if successor split, node is normal
                    idx = nodes[n1].indicator["node_is_normal"] + ind_cntr
                    if idx not in best_effort:
                        cnt_is_normal += 1
                    best_effort[idx] = 1
                else:
                    # otherwise both nodes are normal
                    idx = nodes[n1].indicator["node_is_normal"] + ind_cntr
                    if idx not in best_effort:
                        cnt_is_normal += 1
                    best_effort[idx] = 1
                    idx = nodes[n2].indicator["node_is_normal"] + ind_cntr
                    if idx not in best_effort:
                        cnt_is_normal += 1
                    best_effort[idx] = 1

                # all edges in graph are selected
                idx = edges[(n1, n2)].indicator["edge_selected"] + ind_cntr
                # print(idx)
                best_effort[idx] = 1
                cnt_edge_selected += 1

        ind_cntr += len(feat_block)
        print(ind_cntr)

    # for k,v in feature_blocks.items():
    #     print(k, len(v))

    print(len(best_effort), max(list(best_effort.keys())), ind_cntr)
    # assert (ind_cntr)  % 7 == 0
    # for i in range(0, ind_cntr, 7):
    #     if best_effort[i+1] == 1:
    #         assert best_effort[i] == 1
    #     if best_effort[i+2] == 1:
    #         assert best_effort[i] == 1
    #     if best_effort[i+3] == 1:
    #         assert best_effort[i+4] != 1
    #     if best_effort[i+3] == 1:
    #         assert best_effort[i+5] != 1
    #     if best_effort[i+4] == 1:
    #         assert best_effort[i+5] != 1


    # print(best_effort)
    print(len(best_effort))
    print("cnt_split {}".format(cnt_split))
    print("cnt_selected {}".format(cnt_selected))
    print("cnt_appear {}".format(cnt_appear))
    print("cnt_disappear {}".format(cnt_disappear))
    print("cnt_is_daughter {}".format(cnt_is_daughter))
    print("cnt_is_normal {}".format(cnt_is_normal))
    print("cnt_edge_selected {}".format(cnt_edge_selected))
    print("cnt_edge_split {}".format(cnt_edge_split))

    logger.info("writing best effort...")
    with open("best_effort.txt", 'w') as f:
        for i in range(ind_cntr):
            if i in best_effort:
                f.write("1\n")
            else:
                f.write("0\n")
