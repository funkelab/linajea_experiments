#!/bin/bash

DATADIR="data" # location to store n5s and mongo db files
IMAGE="linajea_ctc.img"
OUTDIR="../Fluo-N3DL-DRO/01_RES" # location for final CTC results 

PATH_TO_TEST="../Fluo-N3DL-DRO" # path to ctc test data folder
GUROBI_LIC="gurobi.lic"  # optional (defaults to Scip solver if no gurobi license file found, but results may vary)

# We use singularity containers to package the dependencies of our method
# You must install singularity (https://singularity.hpcng.org/) to run this script
# Singularity containers can be run by docker, but this alternative is not included here
# This script was tested with singularity version 2.6.1 (https://github.com/hpcng/singularity/releases/tag/2.6.1)
# version 2.6 installation docs are here: https://singularity.hpcng.org/admin-docs/2.6/admin_quickstart.html
SINGULARITY_EXEC="singularity exec -B $DATADIR -B $PWD $IMAGE -B $GUROBI_LIC:/opt/gurobi.lic" # remove gurobi binding if no license file
SINGULARITY_EXEC_NV="singularity exec --nv -B $DATADIR"

mkdir $DATADIR
mkdir $OUTDIR

# step 1: start mongodb server
# the included mongo singularity container has it installed - this command starts it running
# it is more stable to run this in a separate shell and keep it open, but this should work as well
singularity run --bind $DATADIR:/data/db mongo.img --auth &> mongo.log &

# step 1.5 (if necessary) - download data (~30 min)
#./download_and_unzip_ctc_data.sh

# step 2: convert data to n5
$SINGULARITY_EXEC python convert_to_n5.py $DATADIR --test $PATH_TO_TEST -o 1

# step 3: run prediction on GPU, write results to mongo (~2hrs)
$SINGULARITY_EXEC_NV python predict_blockwise.py config_01.toml -i 100000

# step 4: extract edges on CPU (~10 min)
$SINGULARITY_EXEC python extract_edges.py config_01.toml

# step 5: solve ILP using Gurobi (~2hrs) or Scip (much longer)
$SINGULARITY_EXEC python solve.py config_01.toml

# step 6: extract seed points from tiff
$SINGULARITY_EXEC python extract_tracks.py 1 $PATH_TO_TEST

# step 7: filter tracks to seed points
$SINGULARITY_EXEC python track_from_seeds.py config_01.toml -s seeds_test_01.txt

# step 8: segment with watershed and write results to CTC format
$SINGULARITY_EXEC python db_graph_to_ctc.py -c config_01.toml -o $OUTDIR -p selected_from_seeds_1 --frame_begin 0 --frame_end 50 --ws 'setup211_simple_ctc/test01_cell_indicators.n5'  --fg_threshold 0.3 
