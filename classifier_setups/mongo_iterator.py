import numpy as np
import daisy
import logging
import time
from linajea import CandidateDatabase, load_config
import argparse

logger = logging.getLogger(__name__)


def normalize(array):
    logger.debug("automatically normalizing array with dtype=%s", array.dtype)

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
        raise RuntimeError("Automatic normalization for " +
                str(array.dtype) + " not implemented, please "
                "provide a factor.")

    logger.debug("scaling %s with %f", array, factor)
    array = array.astype(dtype)*factor
    return array


class MongoIterator(object):
    def __init__(self,
                 db_host,
                 db_name,
                 data_file,
                 roi_size,
                 n_gpus,
                 gpu_id,
                 n_cpus,
                 cpu_id,
                 fieldname,
                 dataset='raw',
                 transform=None,
                 frames=None,
                 overwrite=False):

        self.db = CandidateDatabase(db_name, db_host)
        self.fieldname = fieldname

        self.data = daisy.open_ds(data_file,
                                  dataset)
        self.voxel_size = self.data.voxel_size
        self.frames = frames
        self.transform = transform
        self.overwrite = overwrite

        self.size = daisy.Coordinate(roi_size) * self.voxel_size

        start = time.time()
        logger.info("Partition DB to workers...")
        logger.debug("Frames: %s", str(frames))
        self.cursor = self.get_cursor(
                n_gpus, n_cpus, gpu_id, cpu_id)
        logger.info(f"...took {time.time() - start} seconds")

    def get_chunks(self, n_elements, k_chunks):
        ch = [(n_elements // k_chunks) +
              (1 if i < (n_elements % k_chunks) else 0)
              for i in range(k_chunks)]
        return ch

    def get_cursor(self, n_gpus, n_cpus, gpu_id, cpu_id):
        query_list = []
        if self.frames:
            query_list.append({'t': {'$lt': self.frames[1]}})
            query_list.append({'t': {'$gte': self.frames[0]}})
        if not self.overwrite:
            query_list.append({self.fieldname: {"$exists": False}})
        query = {'$and': query_list}

        logger.debug("Query: %s" % str(query))
        n_open_documents = self.db.nodes.count_documents(query)
        logger.info("N_open_documents: %d" % n_open_documents)
        gpu_chunks = self.get_chunks(n_open_documents, n_gpus)
        logger.debug(len(gpu_chunks))
        logger.debug(gpu_chunks)
        logger.debug(gpu_id)
        gpu_offset = int(np.sum(gpu_chunks[:gpu_id]))
        gpu_len = gpu_chunks[gpu_id]

        cpu_chunks = self.get_chunks(gpu_len, n_cpus)
        cpu_offset = int(np.sum(cpu_chunks[:cpu_id]))
        cpu_len = cpu_chunks[cpu_id]

        doc_offset = gpu_offset + cpu_offset
        doc_len = cpu_len

        logger.info(f"Partition ({gpu_id}, {cpu_id}): Start {doc_offset}, Len {doc_len}")
        logger.debug("GPU %d %d", gpu_offset, gpu_len)
        logger.debug("CPU %d %d", cpu_offset, cpu_len)
        logger.debug("DOC %d %d", doc_offset, doc_len)
        cursor = self.db.nodes.find(
                query,
                no_cursor_timeout=True).skip(doc_offset).limit(doc_len)
        return cursor

    def __iter__(self):
        return self

    def compute_roi(self, position):
        offset = position - self.size/2
        roi = daisy.Roi(offset, self.size).snap_to_grid(self.voxel_size, mode='closest')
        if roi.get_shape()[0] != self.size[0]:
            roi.set_shape(self.size)
        return roi

    def __next__(self):
        doc = next(self.cursor, None)
        if doc is None:
            logger.info("Cursor has finished: Raising StopIteration")
            raise StopIteration

        logger.debug("Doc: %s" % doc)
        pre_x = int(doc["x"])
        pre_y = int(doc["y"])
        pre_z = int(doc["z"])
        pre_t = int(doc["t"])
        _id = int(doc["id"])
        center = np.array([pre_t, pre_z, pre_y, pre_x])
        logger.debug("Center: %s" % center)
        roi = self.compute_roi(center)
        if self.data.roi.contains(roi):
            array = self.data[roi]
            array.materialize()
            array_data = array.data
            array_data = normalize(array_data)
        else:
            logger.info(
                    "Data roi %s does not contain roi %s: skipping node %s",
                    self.data.roi, roi, _id)
            return self.__next__()

        if self.transform is not None and array_data is not None:
            array_data = self.transform(array_data)
        return {"id": _id, "data": array_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    config = load_config(args.config)
    db_host = config['db_host']
    db_name = config['predict']['candidate_db']
    data_file = config['data_file']
    roi_size = daisy.Coordinate(config['input_shape'])
    mongo_em = MongoIterator(
            db_host,
            db_name,
            data_file,
            roi_size,
            1, 0, 1, 0)
    i = 0
    for doc in mongo_em:
        print("Document: %s" % doc)
        i += 1
        if i > 2: break
