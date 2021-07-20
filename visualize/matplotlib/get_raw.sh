#!/bin/sh

python slices_to_images.py --zarr ../../data/140521/140521.zarr --group raw -t 249 252 -z 300 352 -y 700 1312 2 -x 1000 1612 2 --out mouse_raw_slice.png --outdir ~/mouse_pngs_raw
