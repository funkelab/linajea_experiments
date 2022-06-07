#!/bin/bash

basedir=/nrs/funke/malinmayorc/candidates/120828/
evalside1=eval_side_1
evalside2=eval_side_2
traindir=train
valddir=validation
testdir=test



#training
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside1}_400000 -f 0 451 -g linajea_120828_gt_side_2 -o ${basedir}${evalside1}/${traindir} -m 15 -e 200 250
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside2}_400000 -f 0 451 -g linajea_120828_gt_side_1 -o ${basedir}${evalside2}/${traindir} -m 15 -e 200 250

#validation
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside1}_400000 -f 200 250 -g linajea_120828_gt_side_2 -o ${basedir}${evalside1}/${valddir} -m 15
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside2}_400000 -f 200 250 -g linajea_120828_gt_side_1 -o ${basedir}${evalside2}/${valddir} -m 15

#test
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside1}_400000 -f 0 451 -g linajea_120828_gt_side_1 -o ${basedir}${evalside1}/${testdir} -m 15
python match_candidate_nodes_to_gt.py -c linajea_120828_setup211_simple_${evalside2}_400000 -f 0 451 -g linajea_120828_gt_side_2 -o ${basedir}${evalside2}/${testdir} -m 15
