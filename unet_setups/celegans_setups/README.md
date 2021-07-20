Overview
============

linajea setups for C.elegans data

steps can be controlled via run_celegans.py script, e.g.:

``` shell
    bsub -I -n 4 -gpu num=1:mps=no -R"rusage[mem=25600]" -q slowpoke   python run_celegans.py --config config_celegans.toml --setup_dir experimentName --predict
```

``` shell
    python run_celegans.py --config config_celegans.toml --setup_dir experimentName --no_mknet --no_train --run_predict --run_extract_edges --run_solve --run_evaluate
```
