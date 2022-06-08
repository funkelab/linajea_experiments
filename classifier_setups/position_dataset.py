import torch
import daisy
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


def normalize(array):
    logger.debug("automatically normalizing %s with dtype=%s",
                 array, array.dtype)

    dtype = np.float32

    if array.dtype == np.uint8:
        factor = 1.0/255
    elif array.dtype == np.uint16:
        factor = 1.0/(256*256-1)
    elif array.dtype == np.float32:
        assert array.min() >= 0 and array.max() <= 1, (
                "Values are float but not in [0,1], I don't know how "
                "to normalize. Please provide a factor.")
        factor = 1.0
    else:
        raise RuntimeError(
                "Automatic normalization for " +
                str(array.dtype) + " not implemented, please "
                "provide a factor.")

    logger.debug("scaling %s with %f", array, factor)
    array = array.astype(dtype)*factor
    return array


class PositionDataset(Dataset):
    def __init__(self,
                 positions,
                 data_file,
                 roi_size,
                 dataset='raw'):

        self.positions = positions
        self.data_file = data_file
        self.dataset = dataset
        self.data = daisy.open_ds(data_file,
                                  dataset)
        self.voxel_size = self.data.voxel_size
        self.size = daisy.Coordinate(roi_size) * self.voxel_size
        self.transform = None

    def __getitem__(self, idx):
        position = daisy.Coordinate(self.positions[idx])
        offset = position - self.size/2
        roi = daisy.Roi(offset, self.size).snap_to_grid(
                self.voxel_size, mode='closest')
        if roi.get_shape()[0] != self.size[0]:
            roi.set_shape(self.size)
        array_data = None
        if self.data.roi.contains(roi):
            array = self.data[roi]
            array.materialize()
            array_data = array.data
            array_data = normalize(array_data)
            array_data = self.transform_to_tensor(array_data)
        else:
            logger.info("Data Roi %s does not contain generated roi %s" %
                        (self.data.roi, roi))
        return {"id": idx, "data": array_data}
        return array_data

    def __len__(self):
        return len(self.positions)

    def transform_to_tensor(self, data_array):
        tensor_array = torch.tensor(data_array)
        # Add batch dim:
        tensor_array = tensor_array.unsqueeze(0)
        return tensor_array

    def normalize(self, data_array):
        return self.dataset.normalize(data_array)
