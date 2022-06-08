from torch.utils.data import IterableDataset
import torch
from mongo_iterator import MongoIterator
import argparse
from linajea import load_config
import daisy

import logging

log = logging.getLogger(__name__)


class MongoCandidates(IterableDataset):
    def __init__(self,
                 db_host,
                 db_name,
                 data_file,
                 roi_size,
                 label,
                 n_gpus=1,
                 gpu_id=0,
                 frames=None,
                 overwrite=False):

        log.info(f"Initialize dataset {gpu_id+1}/{n_gpus}...")
        self.db_name = db_name
        self.db_credentials = db_host
        self.dataset = data_file
        self.roi_size = roi_size
        self.label = label
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id
        self.frames = frames
        self.overwrite = overwrite

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            log.info("No worker info available,"
                     " use single process data loading.")
            return MongoIterator(self.db_credentials,
                                 self.db_name,
                                 self.dataset,
                                 self.roi_size,
                                 self.n_gpus,
                                 self.gpu_id,
                                 n_cpus=1,
                                 cpu_id=0,
                                 fieldname=self.label,
                                 transform=self.transform_to_tensor,
                                 frames=self.frames,
                                 overwrite=self.overwrite)
        else:
            n_cpus = int(worker_info.num_workers)
            cpu_id = int(worker_info.id)
            log.info("Worker info available, use multiprocess data loading.")
            log.info(f"Init cpu {cpu_id+1}/{n_cpus}...")
            return MongoIterator(self.db_credentials,
                                 self.db_name,
                                 self.dataset,
                                 self.roi_size,
                                 self.n_gpus,
                                 self.gpu_id,
                                 n_cpus=n_cpus,
                                 cpu_id=cpu_id,
                                 fieldname=self.label,
                                 transform=self.transform_to_tensor,
                                 frames=self.frames,
                                 overwrite=self.overwrite)

    def transform_to_tensor(self, data_array):
        tensor_array = torch.tensor(data_array)
        # Add batch dim:
        tensor_array = tensor_array.unsqueeze(0)
        return tensor_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    config = load_config(args.config)
    db_host = config['db_host']
    db_name = config['predict']['candidate_db']
    data_file = config['data_file']
    roi_size = daisy.Coordinate(config['input_shape'])
    mongo_em = MongoCandidates(
            db_host,
            db_name,
            data_file,
            roi_size,
            2,
            1)

    i = 0
    for doc in mongo_em:
        print(doc)
        i += 1
        if i > 2:
            break
