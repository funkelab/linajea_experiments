# README: Cell Tracking Experiments

The main code base for the linajea cell tracking pipeline is at:
https://www.github.com/funkelab/linajea. This repository contains ground truth, models,
results, and scripts used to run experiments, plus some useful auxiliary
scripts. The purpose is to record the code used to run experiments in past publications.
Future development will occur in the main repository, and this experiments repo
will not be updated for new API changes.

## Accessing data, trained models, and results
Please see [this](https://janelia.figshare.com/collections/Whole-embryo_lineage_reconstruction_with_linajea/7371652) Figshare collection for information on how to access the data, trained models, and tracking results for each dataset used in the Nature Biotech paper.

## Versions of libraries used to run experiments in Nature Biotech paper:
    linajea - most run with branch 1.3-dev, perhaps some bugfixes with 1.3.1-dev. Generally, the v1.3 release should work.
    linajea_experiments - the initial commit of the public repository (v1.3). This uses the old config format.
    singularity containers on /nrs - linajea:v1.3-dev, based on pylb-base:v1.3-dev-updated
See singularity recipes in linajea/singularity/Singularity and linajea/singularity/pylp_base/Singularity at the v1.3 release of the linajea repository for versions of all other dependencies.

## Hyperparameters
The config files in run_script/config_files contain the best parameters obtained from the grid search and used in the Nature Biotech paper.

## Code overview
Here is an overview of this experiments repository by directory - most directories have their own,
more detailed README for more information:

data:
Containing ground truth sparse point annotations for mouse, drosophila, and
zebrafish datasets. This is now a duplicate of what is on Figshare,
but the Figshare also contains (a link to) the actual data, 
trained models, and predictions.

evaluate:
Scripts to get results out of MongoDB and write to CSV for visualization, or
create visualizations from those CSVs. Also for quantitative evaluation of the results in the MongoDB.

linajea_utils:
Useful scripts packaged up for import to other experiments scripts (e.g. create
a foreground mask, remove nodes outwide a mask, write grid search config
files).

run_scripts:
Scripts to run the cell tracking experiments, including prediction, extraction, and ILP solving.

unet_setups:
Scripts to create and train unet models, as well as the configurations and
trained models used in our experiments.

visualize:
Scripts to visualize data in neuroglancer and write tracks into MaMuT xml format.

## Jupytext Note 
Some python files are actually jupyter notebooks that have been converted to plain
python scripts using jupytext.

To view scripts as Jupyter Notebooks:

1) Install jupytext (available on pip, conda)
2) (Optional) Create an .ipynb file by calling `jupytext --to notebook <script>.py`
3) Start a jupyter server and navigate to the file in the web UI
4) Click on the file. It will automatically open the .py file as a notebook if you click on it,
although it will still save as plain text (as opposed to storing the output of cells).
If you chose to create an .ipynb file, it will automatically pair them and keep them in sync.

You can even edit the .py file in a text editor, and then refresh the view to see the changes
without losing your variables.
Turn off autosave in the jupyter view if you want to edit the .py file to avoid getting
out of sync. 
