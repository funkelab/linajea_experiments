from __future__ import print_function
import argparse
import logging
import os
import sys
import time
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("once", category=FutureWarning)

import numpy as np
import torch
from torch.utils import tensorboard
# from torchvision.utils import save_image


from linajea.config import (TrackingConfig,
                            maybe_fix_config_paths_to_machine_and_load)
import torch_model
import torch_loss
from gp_dataloader import GpDataLoader
from utils import (get_latest_checkpoint)

logger = logging.getLogger(__name__)


class Train:
    def __init__(self, config):
        "docstring"

        self.config = config
        # Get the latest checkpoint.
        self.checkpoint_basename = os.path.join(config.general.setup_dir, 'train_net')
        self.latest_checkpoint, self.trained_until = get_latest_checkpoint(
            self.checkpoint_basename)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = torch_model.UnetModelWrapper(self.config, self.trained_until)
        self.model.init_layers()
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to move model to device. If you are using a child process "
                "to run your model, maybe you already initialized CUDA by sending "
                "your model to device in the main process."
            ) from e

        self.input_shape, self.output_shape_1, self.output_shape_2 = \
            self.model.inout_shapes(self.device)
        logger.debug("model: %s", self.model)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        self.optimizer = getattr(torch.optim, self.config.optimizerTorch.optimizer)(
            self.model.parameters(), **self.config.optimizerTorch.get_kwargs())

        self.loss = torch_loss.LossWrapper(
            self.config, current_step=self.trained_until).to(self.device)

        if self.config.train.use_swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, anneal_strategy="linear",
                anneal_epochs=10, swa_lr=self.config.optimizerTorch.kwargs.lr/10.0)

        if self.config.train.use_auto_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=4096.0)

        voxel_size = self.config.train_data.data_sources[0].voxel_size
        input_size = [s*v for (s,v) in zip(self.input_shape, voxel_size)]
        output_size_1 = [s*v for (s,v) in zip(self.output_shape_1, voxel_size)]
        output_size_2 = [s*v for (s,v) in zip(self.output_shape_2, voxel_size)]
        arrays = {"RAW": input_size,
                  "ANCHOR": output_size_2,
                  # "RAW_CROPPED": output_size_2,
                  "GT_PARENT_VECTORS": output_size_1,
                  "GT_CELL_INDICATOR": output_size_1,
                  "GT_CELL_CENTER": output_size_1,
                  "CELL_MASK": output_size_1,
                  # "MAXIMA": output_size_2
                  }
        points = None
        points = {"TRACKS": output_size_1,
                  "CENTER_TRACKS": output_size_2,}
        self.train_loader = GpDataLoader(self.config,
                                         arrays=arrays, points=points,
                                         mode="train")
        self.val_loader = GpDataLoader(self.config,
                                       arrays=arrays, points=points,
                                       mode="val")

        self.summary_writer_train = tensorboard.SummaryWriter(
            os.path.join(self.config.general.setup_dir, "train"))
        self.summary_writer_val = tensorboard.SummaryWriter(
            os.path.join(self.config.general.setup_dir, "val"))

    def train_until(self):
        if self.trained_until >= self.config.train.max_iterations:
            return

        self.maybe_load_checkpoint()

        logger.info("Starting training...")
        batches = self.train_loader.get(device=self.device)
        val_batches = self.val_loader.get(device=self.device)
        prev_scale = None
        for i in range(self.trained_until, self.config.train.max_iterations):
            time_start = time.time()
            batch = next(batches)
            time_loaded = time.time()
            logger.info(
                "Load batch: iteration=%d, time=%f",
                i, time_loaded - time_start)

            loss, ciloss, pvloss, summaries, cisum = self.process_batch(batch)

            if self.config.train.use_auto_mixed_precision:
                self.scaler.scale(loss).backward()

                logger.info("scaler scale %s", self.scaler._scale)
                # if prev_scale != self.scaler._scale:
                for name, param in self.model.named_parameters():
                    # if name == "parent_vectors_batched.layers.0.weight" or\
                    #    name == "cell_indicator_batched.layers.0.weight":
                    #     print(name, torch.flatten(param.grad)[:10], torch.flatten(param)[:10].cpu().detach().numpy())
                        if param.requires_grad:
                            if torch.any(torch.isnan(param.grad)) or\
                               torch.any(torch.isinf(param.grad)):
                                logger.info("nan/inf param %s: %s %s",
                                            name,
                                            torch.flatten(param.grad)[:10],
                                            torch.flatten(param)[:10].cpu().detach().numpy())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                prev_scale = self.scaler._scale
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
            #TODO save checkpoint
            #TODO summaries
            #TODO validation

            self.maybe_save_checkpoint(i+1)
            self.handle_summaries(summaries, i, "train")

            time_processed = time.time()
            logger.info(
                "Process batch: iteration=%d, loss=%f, time=%f",
                i, loss.cpu().detach().numpy(), time_processed - time_loaded)

            if i % self.config.train.val_log_step == 1:
                loss, ciloss, pvloss, summaries, cisum = self.process_batch(
                    next(val_batches))
                self.handle_summaries(summaries, i, "val")
                time_processed = time.time()
                logger.info(
                    "Process Validation batch: iteration=%d, loss=%f, time=%f",
                    i, loss.cpu().detach().numpy(), time_processed - time_loaded)

    def process_batch(self, batch):
        raw = batch['RAW']
        with torch.cuda.amp.autocast(
                enabled=self.config.train.use_auto_mixed_precision):
            outputs = self.model(raw)
            cell_indicator = outputs[0]
            maxima = outputs[1]
            parent_vectors = outputs[3]

            gt_cell_indicator = batch['GT_CELL_INDICATOR']
            gt_cell_center = batch['GT_CELL_CENTER']
            cell_mask = batch['CELL_MASK']
            gt_parent_vectors = batch['GT_PARENT_VECTORS']


            loss, ciloss, pvloss, summaries, cisum = self.loss(
                gt_cell_indicator=gt_cell_indicator,
                cell_indicator=cell_indicator,
                maxima=maxima,
                gt_cell_center=gt_cell_center,
                cell_mask=cell_mask,
                gt_parent_vectors=gt_parent_vectors,
                parent_vectors=parent_vectors)
            assert not (torch.isnan(loss).any() or torch.isinf(loss).any()), "nan in loss!"

        return loss, ciloss, pvloss, summaries, cisum


    def maybe_save_checkpoint(self, iteration):
        if iteration % self.config.train.checkpoint_stride != 0:
            return False

        checkpoint_name = self.checkpoint_basename + '_checkpoint_' + '%i' % iteration
        logger.info("Creating checkpoint %s", checkpoint_name)

        data_to_save ={
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.config.train.use_swa:
            data_to_save['swa_model_state_dict'] = self.swa_model.state_dict()
            data_to_save['swa_scheduler_state_dict'] = self.swa_scheduler.state_dict()
        if self.config.train.use_auto_mixed_precision:
            data_to_save['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(
            data_to_save,
            checkpoint_name,
        )

        return True

    def maybe_load_checkpoint(self):
        if self.latest_checkpoint is None:
            return False
        checkpoint = torch.load(self.latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.config.train.use_swa:
            self.swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
            self.swa_scheduler.load_state_dict(checkpoint["swa_scheduler_state_dict"])
        if self.config.train.use_auto_mixed_precision:
            if "scaler_state_dict" not in checkpoint:
                logger.warning("no scaler state dict in checkpoint!")
            else:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return True

    def handle_summaries(self, summaries, iteration, kind):
        for k, (v, f) in summaries.items():
            if int(iteration) % f != 0:
                continue
            if kind == "val":
                self.summary_writer_val.add_scalar(
                    k, v, iteration)
            else:
                self.summary_writer_train.add_scalar(
                    k, v, iteration)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to config file')
    args = parser.parse_args()

    config = maybe_fix_config_paths_to_machine_and_load(args.config)
    config = TrackingConfig(**config)
    logging.basicConfig(
        level=config.general.logging,
        handlers=[
            logging.FileHandler("run.log", mode='a'),
            logging.StreamHandler(sys.stdout),
            # logging.StreamHandler(sys.stderr)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    train = Train(config)
    train.train_until()
