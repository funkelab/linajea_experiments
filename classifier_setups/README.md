Training Process
================
Prepare the network:
python mknet.py --config <config>.toml

Train:
python train.py --config <config>.toml --iterations <n> --snap_frequency <m>


Create a config file with...
* setup_dir: directory for this setup
* data_file: path to zarr containing raw data
* labeled_candidates: path to file containing candidates labeled with cell
  cycle
* batch_size: number of patches that can fit on the gpu


Notes
=====

* requires gunpowder 017d80093ff80608d2e325a2c3ec18ced5f6d4da
* requires gast==0.2.2 (not sure what that is, but mknet has many warnings otherwise)
