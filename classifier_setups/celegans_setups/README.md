Overview
============
(uses a network similar to vgg, with 'same' or 'valid' convolutions)


specify tasks with `-d`
options:
* 'all'
* 'mknet',
* 'train',
* 'validate_checkpoints',
* 'evaluate',
* 'predict',
* 'match_gt',

``` shell
    run_lsf -c 4 -g 1 python run_lcdc.py -r experiments -c config.toml -d mknet train
```

``` shell
    run_lsf -c 4 -g 1 python run_lcdc.py -id experiments/classifier_YYMMDD_HHMMSS -c experiments/classifier_YYMMDD_HHMMSS/config.toml -d predict match_gt evaluate [--dry_run]
```

- training:
  - flags:
    - use_database: if true, node candidates from a linajea run are used, set training.linajea\_config in config appropriately
- postprocessing:
  - flags:
    - use_database: if true, node candidates from a linajea run are used, set postprocessing.linajea\_config in config appropriately
  - tasks:
    - predict: select all node candidates from some linajea setup database, extract the appropriate window from raw zarr and classify nodes
    - match_gt: probable\_gt\_state, find the closest/best matching gt node for a predicted not, within a certain radius, and add class of that node to db entry (if dry\_run is used, database is not changed, for testing)
    - evaluate: get some evaluation metrics


currently requires this gunpowder branch:
https://github.com/Kainmueller-Lab/gunpowder/tree/v1.1-dev_peter
