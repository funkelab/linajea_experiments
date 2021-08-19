#!/bin/bash

DATADIR="data" # location to store n5s and mongo db files
IMAGE="linajea:v1.3.img"
OUTDIR="ctc_result" # location for final CTC results 
# OUTDIR="../Fluo-N3DL-DRO/01_RES"
PATH_TO_TRAIN="Fluo-N3DL-DRO-train" # path to ctc training data folder
PATH_TO_TEST="Fluo-N3DL-DRO-test" # path to ctc test data folder
GUROBI_LIC="gurobi.lic"  # optional (defaults to Scip solver if no gurobi license file found, but results may vary)

SINGULARITY_EXEC="singularity exec -B $DATADIR -B $PWD $IMAGE -B $GUROBI_LIC:/opt/gurobi.lic" # remove gurobi binding if no license file
SINGULARITY_EXEC_NV="singularity exec --nv -B $DATADIR"

mkdir $DATADIR
mkdir $OUTDIR

# step 1: start mongodb server
# the included mongo singularity container has it installed - this command starts it running
# it is more stable to run this in a separate shell and keep it open, but this should work as well, although may silently fail if already running
singularity run --bind $DATADIR:/data/db mongo.img --auth &> mongo.log &

# step 1.5 (if necessary) - download data (~30 min)
#./download_and_unzip_ctc_data.sh

# step 2: convert data to n5
$SINGULARITY_EXEC python convert_to_n5.py $PATH_TO_TRAIN $PATH_TO_TEST $DATADIR -o 2

# step 3: run prediction on GPU, write results to mongo (~2hrs)
$SINGULARITY_EXEC_NV python predict_blockwise.py config_02.toml -i 100000

# step 4: extract edges on CPU (~10 min)
$SINGULARITY_EXEC python extract_edges.py config_02.toml

# step 5: solve ILP using Gurobi (~2hrs) or Scip (much longer)
$SINGULARITY_EXEC python solve.py config_02.toml

# step 6: extract seed points from tiff
$SINGULARITY_EXEC python extract_tracks.py 2 $PATH_TO_TEST

# step 7: filter tracks to seed points
$SINGULARITY_EXEC python track_from_seeds.py config_02.toml -s seeds_test_02.txt

# segment with watershed and write results to CTC format
$SINGULARITY_EXEC python db_graph_to_ctc.py -c config_02.toml -o $OUTDIR -p selected_from_seeds_1 --frame_begin 0 --frame_end 50 --ws 'setup211_simple_ctc/test02_cell_indicators.n5'  --fg_threshold 0.3 
