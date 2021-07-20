#!/bin/sh

python slices_to_images.py --zarr ../../unet_setups/setup11_simple_train_early/predictions_245_255.zarr --group volumes/cell_indicator -t 4 7 -z 300 352 -y 700 1312 2 -x 1000 1612 2 --out mouse_ci_slice.png --outdir /nrs/funke/malinmayorc/mouse_pngs_cell_indicator --norm range --range 0 1 --color
