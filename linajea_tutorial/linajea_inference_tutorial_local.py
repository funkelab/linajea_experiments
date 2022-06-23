# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Example Pipeline

# %load_ext autoreload
# %autoreload 2

from daisy import Roi, open_ds

# # Load and visualize data

path_to_raw = '140521_250_1500_700_1100.n5'
raw_ds_name = 'raw'

raw_ds = open_ds(path_to_raw, raw_ds_name)

total_roi = raw_ds.roi
sample_data_array = raw_ds[total_roi].to_ndarray()

import matplotlib.pyplot as plt
# show first time point, first z-slice
plt.imshow(sample_data_array[0][0])
plt.show()

# # Run inference with pre-trained network

from linajea.process_blockwise import predict_blockwise
from linajea.config import TrackingConfig

import logging
logging.basicConfig(level=logging.INFO)

linajea_config = TrackingConfig.from_file('tutorial_tracking_config.toml')

predict_blockwise(linajea_config) # this will take a number of hours

# ## Train Cell Indicator and Parent Vector Networks

tracks = viewer.add_tracks(tracks_data, name='gt_tracks')

points = viewer.add_points(tracks_data[:,1:], name='points_tracks')


