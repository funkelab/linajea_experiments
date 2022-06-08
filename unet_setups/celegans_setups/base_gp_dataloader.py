from __future__ import print_function
import logging
import warnings
warnings.filterwarnings("once", category=FutureWarning)

import torch
import gunpowder as gp

logger = logging.getLogger(__name__)


class BaseGpDataLoader:
    def __init__(self, config, *, mode, arrays=None, points=None):
        "docstring"

        self.config = config
        self.mode = mode
        self.pipeline = None

        self.arrays = arrays
        self.array_keys = {}
        if arrays is not None:
            for k, s in self.arrays.items():
                self.array_keys[k] = gp.ArrayKey(k)

        self.points = points
        self.graph_keys = {}
        if points is not None:
            for k, s in points.items():
                self.graph_keys[k] = gp.GraphKey(k)

        self.request = gp.BatchRequest()
        if arrays is not None:
            for n, k in self.array_keys.items():
                self.request.add(k, self.arrays[n])

        if points is not None:
            for n, k in self.graph_keys.items():
                self.request.add(k, self.points[n])

        logger.info("REQUEST: %s" % str(self.request))


    def add_inputs(self):
        return None

    def add_pre_aug_ops(self):
        return None

    def add_aug_ops(self):
        return None

    def add_post_aug_ops(self):
        return None

    def assemble_pipeline(self):
        logger.info("assembling pipeline")
        if self.pipeline is None:
            logger.info("adding inputs")
            self.pipeline = self.add_inputs()
            pre_aug_ops = self.add_pre_aug_ops()
            if pre_aug_ops is not None:
                logger.info("adding pre aug ops")
                self.pipeline += pre_aug_ops
            else:
                logger.info("skipping pre aug ops")
            if self.mode == "train":
                aug_ops = self.add_aug_ops()
                if aug_ops is not None:
                    logger.info("adding aug ops")
                    self.pipeline += aug_ops
                else:
                    logger.info("skipping aug ops")
            post_aug_ops = self.add_post_aug_ops()
            if post_aug_ops is not None:
                logger.info("adding post aug ops")
                self.pipeline += post_aug_ops
            else:
                logger.info("skipping post aug ops")
        logger.info("pipeline assembly done")

    def get(self, device):
        if self.pipeline is None:
            self.assemble_pipeline()

        with gp.build(self.pipeline):

            logger.info("Starting data loading...")
            while True:
                batchT = self.pipeline.request_batch(self.request)
                # TODO setup dict
                batch = {}
                for k, v in batchT.items():
                    if "anchor" in k.identifier.lower():
                        continue
                    if isinstance(k, gp.GraphKey):
                        batch[k.identifier] = v
                    if isinstance(k, gp.ArrayKey):
                        batch[k.identifier] = v.data
                        if device is not None:
                            batch[k.identifier] = torch.as_tensor(
                                batch[k.identifier], device=device)
                yield batch
