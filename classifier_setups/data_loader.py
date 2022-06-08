import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def get_data_loader(
        torch_dataset,
        batch_size,
        roi_size,
        num_workers=5,
        prefetch_factor=40,
        persistent_workers=False,
        ):
    logger.debug("Creating pytorch data loader with %d workers,"
                 " %d batch_size, %s roi_size", num_workers, batch_size,
                 str(roi_size))

    def collate_fn(batch):
        while len(batch) < batch_size:
            tensor_array = torch.tensor(np.zeros(roi_size, dtype=np.float32))
            # Add batch dim:
            tensor_array = tensor_array.unsqueeze(0)
            batch.append({'id': None, 'data': tensor_array})

        batch_data = {"id": [b["id"] for b in batch],
                      "data": torch.cat([b["data"] for b in batch], dim=0)
                      }
        return batch_data

    loader = DataLoader(torch_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                        pin_memory=True)
    return loader
