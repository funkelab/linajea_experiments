from __future__ import print_function
import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("once", category=FutureWarning)
# warnings to errors:
# warnings.filterwarnings("error")

import argparse
import glob
import re
import json
import time
import toml
import zarr
from linajea.gunpowder import (TracksSource, AddParentVectors,
                               ShiftAugment, Clip, NoOp,
                               NormalizeMinMax, NormalizeMeanStd,
                               NormalizeMedianMad)
from linajea.config import (TrackingConfig,
                            maybe_fix_config_paths_to_machine_and_load)
from linajea import load_config

import gunpowder as gp
import logging
import math
import os
import sys
import torch
from torchvision.utils import save_image
import numpy as np



import torch_model

# try:
#     import absl.logging
#     logging.root.removeHandler(absl.logging._absl_handler)
#     absl.logging._warn_preinit_stderr = False
# except Exception as e:
#     print(e)


logger = logging.getLogger(__name__)

class Cast(gp.BatchFilter):

    def __init__(
            self,
            array,
            dtype=np.float32):

        self.array = array
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        array_spec = copy.deepcopy(self.spec[self.array])
        array_spec.dtype = self.dtype
        self.updates(self.array, array_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array]
        deps[self.array].dtype = None
        return deps

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.spec.dtype = self.dtype
        array.data = array.data.astype(self.dtype)



class LossWrapper(torch.nn.Module):
    def __init__(self, config, current_step=0):
        super().__init__()
        self.config = config
        self.voxel_size = torch.nn.Parameter(
            torch.FloatTensor(self.config.train_data.voxel_size[1:]),
            requires_grad=False)
        self.current_step = torch.nn.Parameter(
            torch.tensor(float(current_step)), requires_grad=False)

        def weighted_mse_loss(inputs, target, weight):
            # print(((inputs - target) ** 2).size())
            # print((weight * ((inputs - target) ** 2)).size())
            # return (weight * ((inputs - target) ** 2)).mean()
            # print((weight * ((inputs - target) ** 2)).sum(), weight.sum())
            ws = weight.sum() * inputs.size()[0] / weight.size()[0]
            if abs(ws) <= 0.001:
                ws = 1
            return (weight * ((inputs - target) ** 2)).sum()/ ws

        def weighted_mse_loss2(inputs, target, weight):
            # print(((inputs - target) ** 2).size())
            # print((weight * ((inputs - target) ** 2)).size())
            # return (weight * ((inputs - target) ** 2)).mean()
            # print((weight * ((inputs - target) ** 2)).sum(), weight.sum())
            # ws = weight.sum()
            # if ws == 0:
                # ws = 1
            return (weight * ((inputs - target) ** 2)).mean()

        # self.pv_loss = torch.nn.MSELoss(reduction='mean')
        # self.ci_loss = torch.nn.MSELoss(reduction='mean')
        self.pv_loss = weighted_mse_loss
        self.ci_loss = weighted_mse_loss2

        met_sum_intv = 10
        loss_sum_intv = 1
        self.summaries = {
            "loss":                          [-1, loss_sum_intv],
            "alpha":                         [-1, loss_sum_intv],
            "cell_indicator_loss":           [-1, loss_sum_intv],
            "parent_vectors_loss_cell_mask": [-1, loss_sum_intv],
            "parent_vectors_loss_maxima":    [-1, loss_sum_intv],
            "parent_vectors_loss":           [-1, loss_sum_intv],
            'cell_ind_tpr_gt':               [-1, met_sum_intv],
            'par_vec_cos_gt':                [-1, met_sum_intv],
            'par_vec_diff_mn_gt':            [-1, met_sum_intv],
            'par_vec_tpr_gt':                [-1, met_sum_intv],
            'cell_ind_tpr_pred':             [-1, met_sum_intv],
            'par_vec_cos_pred':              [-1, met_sum_intv],
            'par_vec_diff_mn_pred':          [-1, met_sum_intv],
            'par_vec_tpr_pred':              [-1, met_sum_intv],
            }

    def metric_summaries(self,
                         gt_cell_center,
                         cell_indicator,
                         cell_indicator_cropped,
                         gt_parent_vectors_cropped,
                         parent_vectors_cropped,
                         maxima,
                         maxima_in_cell_mask,
                         output_shape_2):


        # ground truth cell locations
        gt_max_loc = torch.nonzero(gt_cell_center > 0.5)

        # predicted value at those locations
        tmp = cell_indicator[list(gt_max_loc.T)]
        # tmp = tf.gather_nd(cell_indicator, gt_max_loc)
        # true positive if > 0.5
        cell_ind_tpr_gt = torch.mean((tmp > 0.5).float())

        if not self.config.model.train_only_cell_indicator:
            # crop to nms area
            gt_cell_center_cropped = torch_model.crop(
                # l=1, d, h, w
                gt_cell_center,
                # l=1, d', h', w'
                output_shape_2)
            tp_dims = [1, 2, 3, 4, 0]

            # cropped ground truth cell locations
            gt_max_loc = torch.nonzero(gt_cell_center_cropped > 0.5)

            # ground truth parent vectors at those locations
            tmp_gt_par = gt_parent_vectors_cropped.permute(*tp_dims)[list(gt_max_loc.T)]

            # predicted parent vectors at those locations
            tmp_par = parent_vectors_cropped.permute(*tp_dims)[list(gt_max_loc.T)]

            # normalize predicted parent vectors
            normalize_pred = torch.nn.functional.normalize(tmp_par, dim=1)
            # normalize ground truth parent vectors
            normalize_gt = torch.nn.functional.normalize(tmp_gt_par, dim=1)

            # cosine similarity predicted vs ground truth parent vectors
            cos_similarity = torch.sum(normalize_pred * normalize_gt,
                                       dim=1)

            # rate with cosine similarity > 0.9
            par_vec_cos_gt = torch.mean((cos_similarity > 0.9).float())

            # distance between endpoints of predicted vs gt parent vectors
            par_vec_diff = torch.linalg.vector_norm(
                (tmp_gt_par / self.voxel_size) - (tmp_par / self.voxel_size), dim=1)
            # mean distance
            par_vec_diff_mn_gt = torch.mean(par_vec_diff)

            # rate with distance < 1
            par_vec_tpr_gt = torch.mean((par_vec_diff < 1).float())

        # predicted cell locations
        # pred_max_loc = torch.nonzero(torch.reshape(maxima_in_cell_mask, output_shape_2))
        pred_max_loc = torch.nonzero(torch.reshape(
            torch.gt(maxima * cell_indicator_cropped, 0.2), output_shape_2))

        # predicted value at those locations
        tmp = cell_indicator_cropped[list(pred_max_loc.T)]
        # assumed good if > 0.5
        cell_ind_tpr_pred = torch.mean((tmp > 0.5).float())

        if not self.config.model.train_only_cell_indicator:
            tp_dims = [1, 2, 3, 4, 0]
            # ground truth parent vectors at those locations
            tmp_gt_par = gt_parent_vectors_cropped.permute(*tp_dims)[list(pred_max_loc.T)]

            # predicted parent vectors at those locations
            tmp_par = parent_vectors_cropped.permute(*tp_dims)[list(pred_max_loc.T)]

            # normalize predicted parent vectors
            normalize_pred = torch.nn.functional.normalize(tmp_par, dim=1)

            # normalize ground truth parent vectors
            normalize_gt = torch.nn.functional.normalize(tmp_gt_par, dim=1)

            # cosine similarity predicted vs ground truth parent vectors
            cos_similarity = torch.sum(normalize_pred * normalize_gt,
                                       dim=1)

            # rate with cosine similarity > 0.9
            par_vec_cos_pred = torch.mean((cos_similarity > 0.9).float())

            # distance between endpoints of predicted vs gt parent vectors
            par_vec_diff = torch.linalg.vector_norm(
                (tmp_gt_par / self.voxel_size) - (tmp_par / self.voxel_size), dim=1)

            # mean distance
            par_vec_diff_mn_pred = torch.mean(par_vec_diff)

            # rate with distance < 1
            par_vec_tpr_pred = torch.mean((par_vec_diff < 1).float())

        self.summaries['cell_ind_tpr_gt'][0] = cell_ind_tpr_gt
        self.summaries['cell_ind_tpr_pred'][0] = cell_ind_tpr_pred
        if not self.config.model.train_only_cell_indicator:
            self.summaries['par_vec_cos_gt'][0] = par_vec_cos_gt
            self.summaries['par_vec_diff_mn_gt'][0] = par_vec_diff_mn_gt
            self.summaries['par_vec_tpr_gt'][0] = par_vec_tpr_gt
            self.summaries['par_vec_cos_pred'][0] = par_vec_cos_pred
            self.summaries['par_vec_diff_mn_pred'][0] = par_vec_diff_mn_pred
            self.summaries['par_vec_tpr_pred'][0] = par_vec_tpr_pred


    def forward(self, *,
                gt_cell_indicator,
                cell_indicator,
                maxima,
                gt_cell_center,
                cell_mask=None,
                gt_parent_vectors=None,
                parent_vectors=None
                ):

        output_shape_1 = cell_indicator.size()
        output_shape_2 = maxima.size()

        # raw_cropped = torch_model.crop(raw, output_shape_1)
        cell_indicator_cropped = torch_model.crop(
            # l=1, d, h, w
            cell_indicator,
            # l=1, d', h', w'
            output_shape_2)

        if not self.config.model.train_only_cell_indicator:
            cell_mask = torch.reshape(cell_mask, (1,) + output_shape_1)
            # l=1, d', h', w'
            cell_mask_cropped = torch_model.crop(
                # l=1, d, h, w
                cell_mask,
                # l=1, d', h', w'
                output_shape_2)

            # print(maxima.size(), cell_indicator_cropped.size(), cell_indicator.size())
            # maxima = torch.eq(maxima, cell_indicator_cropped)

            # l=1, d', h', w'
            # maxima_in_cell_mask = torch.logical_and(maxima, cell_mask_cropped)
            maxima_in_cell_mask = maxima.float() * cell_mask_cropped.float()
            maxima_in_cell_mask = torch.reshape(maxima_in_cell_mask,
                                                (1,) + output_shape_2)

            # c=3, l=1, d', h', w'
            parent_vectors_cropped = torch_model.crop(
                # c=3, l=1, d, h, w
                parent_vectors,
                # c=3, l=1, d', h', w'
                (3,) + output_shape_2)

            # c=3, l=1, d', h', w'
            gt_parent_vectors_cropped = torch_model.crop(
                # c=3, l=1, d, h, w
                gt_parent_vectors,
                # c=3, l=1, d', h', w'
                (3,) + output_shape_2)

            parent_vectors_loss_cell_mask = self.pv_loss(
                # c=3, l=1, d, h, w
                gt_parent_vectors,
                # c=3, l=1, d, h, w
                parent_vectors,
                # c=1, l=1, d, h, w (broadcastable)
                cell_mask)

            # cropped
            parent_vectors_loss_maxima = self.pv_loss(
                # c=3, l=1, d', h', w'
                gt_parent_vectors_cropped,
                # c=3, l=1, d', h', w'
                parent_vectors_cropped,
                # c=1, l=1, d', h', w' (broadcastable)
                torch.reshape(maxima_in_cell_mask, (1,) + output_shape_2))
        else:
            parent_vectors_cropped = None
            gt_parent_vectors_cropped = None
            maxima_in_cell_mask = None

        # non-cropped
        if self.config.model.cell_indicator_weighted:
            if isinstance(self.config.model.cell_indicator_weighted, bool):
                self.config.model.cell_indicator_weighted = 0.00001
            cond = gt_cell_indicator < self.config.model.cell_indicator_cutoff
            weight = torch.where(cond, self.config.model.cell_indicator_weighted, 1.0)
            # print(cond.size(), weight.size())
        else:
            weight = torch.tensor(1.0)
        # print(torch.min(cell_indicator), torch.max(cell_indicator))
        # print(torch.min(gt_cell_indicator), torch.max(gt_cell_indicator))
        # for i, w in enumerate(torch.squeeze(weight, dim=0)):
        #     save_image(w, 'w_{}_{}.tif'.format(self.current_step, i))
        # for i, ci in enumerate(torch.squeeze(cell_indicator, dim=0)):
        #     save_image(ci, 'ci_{}_{}.tif'.format(self.current_step, i))
        # print(np.min(cell_indicator.detach().cpu().numpy()),
        #       np.max(cell_indicator.detach().cpu().numpy()))
        cell_indicator_loss = self.ci_loss(
            # l=1, d, h, w
            gt_cell_indicator,
            # l=1, d, h, w
            cell_indicator,
            # l=1, d, h, w
            weight)
            # cell_mask)

        # print(torch.sum(gt_cell_indicator), torch.sum(cell_indicator), torch.sum(weight))
        if self.config.model.train_only_cell_indicator:
            loss = cell_indicator_loss
        else:
            if self.config.train.parent_vectors_loss_transition_offset:
                # smooth transition from training parent vectors on complete cell mask to
                # only on maxima
                # https://www.wolframalpha.com/input/?i=1.0%2F(1.0+%2B+exp(0.01*(-x%2B20000)))+x%3D0+to+40000
                alpha = (1.0 /
                         (1.0 + torch.exp(
                             config.train.parent_vectors_loss_transition_factor *
                             (-self.current_step + config.train.parent_vectors_loss_transition_offset))))
                self.summaries['alpha'][0] = alpha

                # multiply cell indicator loss with parent vector loss, since they have
                # different magnitudes (this normalizes gradients by magnitude of other
                # loss: (uv)' = u'v + uv')
                # loss = 1000 * cell_indicator_loss +  0.1 * (
                #     parent_vectors_loss_maxima*alpha +
                #     parent_vectors_loss_cell_mask*(1.0 - alpha)
                # )
                parent_vectors_loss = (parent_vectors_loss_maxima * alpha +
                                       parent_vectors_loss_cell_mask * (1.0 - alpha)
                                       )
            else:
                # loss = 1000 * cell_indicator_loss + 0.1 * parent_vectors_loss_cell_mask
                parent_vectors_loss = parent_vectors_loss_cell_mask

            loss = cell_indicator_loss + parent_vectors_loss
            self.summaries['parent_vectors_loss_cell_mask'][0] = parent_vectors_loss_cell_mask
            self.summaries['parent_vectors_loss_maxima'][0] = parent_vectors_loss_maxima
            self.summaries['parent_vectors_loss'][0] = parent_vectors_loss
        # print(loss, cell_indicator_loss, parent_vectors_loss)

        self.summaries['loss'][0] = loss
        self.summaries['cell_indicator_loss'][0] = cell_indicator_loss

        self.metric_summaries(
            gt_cell_center,
            cell_indicator,
            cell_indicator_cropped,
            gt_parent_vectors_cropped,
            parent_vectors_cropped,
            maxima,
            maxima_in_cell_mask,
            output_shape_2)

        self.current_step += 1
        return loss, cell_indicator_loss, parent_vectors_loss, self.summaries, torch.sum(cell_indicator_cropped)


def get_latest_checkpoint(basename):

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    checkpoints = glob.glob(basename + '_checkpoint_*')
    checkpoints.sort(key=natural_keys)

    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        iteration = int(checkpoint.split('_')[-1].split('.')[0])
        return checkpoint, iteration

    return None, 0


def merge_sources(
        raw,
        tracks,
        center_tracks,
        track_file,
        center_cell_file,
        limit_to_roi,
        scale=1.0,
        use_radius=False
        ):
    return (
        (raw,
         # tracks
         TracksSource(
                track_file,
             tracks,
             points_spec=gp.PointsSpec(roi=limit_to_roi),
             scale=scale,
             use_radius=use_radius),
         # center tracks
         TracksSource(
             center_cell_file,
             center_tracks,
             points_spec=gp.PointsSpec(roi=limit_to_roi),
             scale=scale,
             use_radius=use_radius),
         ) +
        gp.MergeProvider() +
        # not None padding works in combination with ensure_nonempty in
        # random_location as always a random point is picked and the roi
        # shifted such that that point is inside
        gp.Pad(tracks, gp.Coordinate((0, 500, 500, 500))) +
        gp.Pad(center_tracks, gp.Coordinate((0, 500, 500, 500)))
        # gp.Pad(tracks, None) +
        # gp.Pad(center_tracks, None)
    )


def train_until(config):
    # Get the latest checkpoint.
    checkpoint_basename = os.path.join(config.general.setup_dir, 'train_net')
    latest_checkpoint, trained_until = get_latest_checkpoint(checkpoint_basename)
    if trained_until >= config.train.max_iterations:
        return

    # net_config = load_config(os.path.join(config.general.setup_dir,
    #                                       'train_net_config.json'))
    # net_names = load_config(os.path.join(config.general.setup_dir,
    #                                      'train_net_names.json'))
    # if 'summaries' in net_names:
    #     summaries = net_names['summaries']
    # else:
    #     summaries = None
    #     for k, v in net_names.items():
    #         if "summaries" in k:
    #             if summaries is None:
    #                 summaries = {}
    #             if "scalar" in k:
    #                 summaries[k] = (v, 1)
    #                 if config.train.val_log_step is not None:
    #                     summaries["validation_" + k] = (v, config.train.val_log_step)
    #             elif "network" in k:
    #                 summaries[k] = (v, 25)
    #             elif "histo" in k:
    #                 summaries[k] = (v, 50)
    #             elif "image" in k:
    #                 summaries[k] = (v, 50)
    #             elif "metric" in k:
    #                 summaries[k] = (v, 10)
    #                 if config.train.val_log_step is not None:
    #                     summaries["validation_" + k] = (v, config.train.val_log_step)
    #             else:
    #                 summaries[k] = (v, 10)


    anchor = gp.ArrayKey('ANCHOR')
    raw = gp.ArrayKey('RAW')
    # raw_tmp = gp.ArrayKey('RAW_TMP')
    raw_cropped = gp.ArrayKey('RAW_CROPPED')
    tracks = gp.PointsKey('TRACKS')
    center_tracks = gp.PointsKey('CENTER_TRACKS')
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    cell_center = gp.ArrayKey('CELL_CENTER')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    maxima = gp.ArrayKey('MAXIMA')
    # grad_cell_indicator = gp.ArrayKey('GRAD_CELL_INDICATOR')
    if not config.model.train_only_cell_indicator:
        parent_vectors = gp.ArrayKey('PARENT_VECTORS')
        pred_parent_vectors = gp.ArrayKey('PRED_PARENT_VECTORS')
        cell_mask = gp.ArrayKey('CELL_MASK')
        # grad_parent_vectors = gp.ArrayKey('GRAD_PARENT_VECTORS')
    if config.train.cell_density:
        cell_density = gp.ArrayKey('CELL_DENSITY')
        pred_cell_density = gp.ArrayKey('PRED_CELL_DENSITY')
    else:
        cell_density = None

    torch.backends.cudnn.benchmark = True
    model = torch_model.UnetModelWrapper(config, trained_until)
    print(model)
    # the default init in pytorch is a bit strange
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
    # some modified (for backwards comp) version of kaiming
    # breaks training, cell_ind -> 0
    # activation func relu
    def init_weights(m):
        # print("init", m)
        if isinstance(m, torch.nn.Conv3d):
            # print("init")
            # torch.nn.init.xavier_uniform(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
    # activation func sigmoid
    def init_weights_sig(m):
        if isinstance(m, torch.nn.Conv3d):
            # torch.nn.init.xavier_uniform(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    logger.info("initializing model..")
    model.apply(init_weights)
    model.cell_indicator_batched.apply(init_weights_sig)
    if not config.model.train_only_cell_indicator:
        model.parent_vectors_batched.apply(init_weights)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e
    logger.info("getting train/test output shape by running model twice")
    input_shape_predict = config.model.predict_input_shape
    model.eval()
    with torch.no_grad():
        trial_run_predict = model.forward(torch.zeros(input_shape_predict,
                                                      dtype=torch.float32).to(device))
    model.train()
    logger.info("test done")
    if not config.model.train_only_cell_indicator:
        _, _, trial_max_predict, _ = trial_run_predict
    else:
        _, _, trial_max_predict = trial_run_predict
    output_shape_predict = trial_max_predict.size()
    net_config = {
        'input_shape': input_shape_predict,
        'output_shape_2': output_shape_predict
    }
    with open(os.path.join(config.general.setup_dir, 'test_net_config.json'), 'w') as f:
        json.dump(net_config, f)

    input_shape = config.model.train_input_shape
    trial_run = model.forward(torch.zeros(input_shape, dtype=torch.float32).to(device))
    if not config.model.train_only_cell_indicator:
        trial_ci, _, trial_max, _ = trial_run
    else:
        trial_ci, _, trial_max = trial_run
    output_shape_1 = trial_ci.size()
    output_shape_2 = trial_max.size()
    logger.info("train done")
    net_config = {
        'input_shape': input_shape,
        'output_shape_1': output_shape_1,
        'output_shape_2': output_shape_2
    }
    with open(os.path.join(config.general.setup_dir, 'train_net_config.json'), 'w') as f:
        json.dump(net_config, f)

    voxel_size = gp.Coordinate(config.train_data.data_sources[0].voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size_1 = gp.Coordinate(output_shape_1) * voxel_size
    output_size_2 = gp.Coordinate(output_shape_2) * voxel_size
    # center = [max(1, int(d * 0.75)) for d in net_config['output_shape_2']]
    center_size = gp.Coordinate(output_shape_2) * voxel_size
    # add a buffer in time to avoid choosing a random location based on points
    # in only one frame, because then all points are rejected as being on the
    # lower boundary of that frame
    center_size = center_size + gp.Coordinate((1, 0, 0, 0))
    logger.info("Center size: {}".format(center_size))
    logger.info("Output size 1: {}".format(output_size_1))
    logger.info("Voxel size: {}".format(voxel_size))

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(tracks, output_size_1)
    request.add(center_tracks, center_size)
    request.add(cell_indicator, output_size_1)
    request.add(cell_center, output_size_1)
    request.add(anchor, output_size_2)
    # request.add(anchor, input_size)
    request.add(raw_cropped, output_size_2)
    request.add(maxima, output_size_2)
    if not config.model.train_only_cell_indicator:
        request.add(parent_vectors, output_size_1)
        request.add(cell_mask, output_size_1)
    logger.info("REQUEST: %s" % str(request))
    snapshot_request = gp.BatchRequest({
        raw: request[raw],
        # raw_cropped: request[parent_vectors],
        # pred_cell_indicator: request[parent_vectors],
        # grad_parent_vectors: request[parent_vectors],
        # grad_cell_indicator: request[parent_vectors]
    })
    snapshot_request.add(pred_cell_indicator, output_size_1)
    snapshot_request.add(raw_cropped, output_size_2)
    snapshot_request.add(maxima, output_size_2)
    if not config.model.train_only_cell_indicator:
        snapshot_request.add(pred_parent_vectors, output_size_1)
    # snapshot_request.add(raw_tmp, input_size)
    if config.train.cell_density:
        request.add(cell_density, output_size_1)
        snapshot_request.add(cell_density, output_size_1)
        snapshot_request.add(pred_cell_density, output_size_1)
    logger.info("Snapshot request: %s" % str(snapshot_request))


    train_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                config.train_data.data_sources,
                                cell_density=cell_density, val=False)
    if config.train.val_log_step is not None:
        val_sources = get_sources(config, raw, anchor, tracks, center_tracks,
                                  config.validate_data.data_sources,
                                  cell_density=cell_density, val=True)

    augment = config.train.augment
    train_pipeline = (
        tuple(train_sources) +
        gp.RandomProvider() +

        (gp.ElasticAugment(
            augment.elastic.control_point_spacing,
            augment.elastic.jitter_sigma,
            [augment.elastic.rotation_min*np.pi/180.0,
             augment.elastic.rotation_max*np.pi/180.0],
            rotation_3d=augment.elastic.rotation_3d,
            subsample=augment.elastic.subsample,
            use_fast_points_transform=augment.elastic.use_fast_points_transform,
            spatial_dims=3,
            temporal_dim=True) \
         if augment.elastic is not None else NoOp()) +

        (ShiftAugment(
            prob_slip=augment.shift.prob_slip,
            prob_shift=augment.shift.prob_shift,
            sigma=augment.shift.sigma,
            shift_axis=0) \
         if augment.shift is not None else NoOp()) +

        (gp.SimpleAugment(
            mirror_only=augment.simple.mirror,
            transpose_only=augment.simple.transpose) \
         if augment.simple is not None else NoOp()) +

        (gp.ZoomAugment(
            factor_min=augment.zoom.factor_min,
            factor_max=augment.zoom.factor_max,
            spatial_dims=augment.zoom.spatial_dims,
            order={raw: 1,
                   }) \
         if augment.zoom is not None else NoOp()) +


        (gp.NoiseAugment(
            raw,
            mode='gaussian',
            var=augment.noise_gaussian.var,
            clip=False) \
         if augment.noise_gaussian is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='speckle',
            var=augment.noise_speckle.var,
            clip=False) \
         if augment.noise_speckle is not None else NoOp()) +

        (gp.NoiseAugment(
            raw,
            mode='s&p',
            amount=augment.noise_saltpepper.amount,
            clip=False) \
         if augment.noise_saltpepper is not None else NoOp()) +

        (gp.HistogramAugment(
            raw,
            # raw_tmp,
            range_low=augment.histogram.range_low,
            range_high=augment.histogram.range_high,
            z_section_wise=False) \
        if augment.histogram is not None else NoOp())  +

        (gp.IntensityAugment(
            raw,
            scale_min=augment.intensity.scale[0],
            scale_max=augment.intensity.scale[1],
            shift_min=augment.intensity.shift[0],
            shift_max=augment.intensity.shift[1],
            z_section_wise=False,
            clip=False) \
         if augment.intensity is not None else NoOp()) +

        (AddParentVectors(
            tracks,
            parent_vectors,
            cell_mask,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            radius=config.train.parent_radius,
            move_radius=config.train.move_radius,
            dense=not config.general.sparse) \
         if not config.model.train_only_cell_indicator else NoOp()) +

        # not used because spare = false in celegans
        (gp.Reject(
            mask=cell_mask,
            min_masked=0.0001,
            # reject_probability=augment.reject_empty_prob
        ) \
         if config.general.sparse else NoOp()) +

        # already done by random_location ensure_nonempty arg
        # no, not done, elastic augment changes roi, point might
        # be in outer part that gets cropped
        # but that's ok
        # gp.Reject(
        #     # be in outer part that gets cropped
        #     ensure_nonempty=center_tracks,
        #     reject_probability=augment.reject_empty_prob
        # ) +

        gp.RasterizeGraph(
            tracks,
            cell_indicator,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=config.train.rasterize_radius,
                mode='peak')) +

        gp.RasterizeGraph(
            tracks,
            cell_center,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.1,) + voxel_size[1:],
                mode='point'))
        +

        (gp.PreCache(
            cache_size=config.train.cache_size,
            num_workers=config.train.job.num_workers) \
         if config.train.job.num_workers > 1 else NoOp())
    )

    if config.train.val_log_step is not None:
        val_pipeline = (
            tuple(val_sources) +
            gp.RandomProvider() +

            (AddParentVectors(
                tracks,
                parent_vectors,
                cell_mask,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                radius=config.train.parent_radius,
                move_radius=config.train.move_radius,
                dense=not config.general.sparse) \
             if not config.model.train_only_cell_indicator else NoOp()) +

            (gp.Reject(
                mask=cell_mask,
                min_masked=0.0001,
                # reject_probability=augment.reject_empty_prob
            ) \
             if config.general.sparse else NoOp()) +

            gp.Reject(
                # already done by random_location ensure_nonempty arg
                # no, not done, elastic augment changes roi, point might
                # be in outer part that gets cropped
                ensure_nonempty=center_tracks,
                # always reject emtpy batches in validation branch
                # reject_probability=augment.reject_empty_prob
            ) +

            gp.RasterizePoints(
                tracks,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=config.train.rasterize_radius,
                    mode='peak')) +

            gp.RasterizePoints(
                tracks,
                cell_center,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=(0.1,) + voxel_size[1:],
                    mode='point'))
            +

            (gp.PreCache(
                cache_size=config.train.cache_size,
                num_workers=1) \
             if config.train.job.num_workers > 1 else NoOp())
        )

    if config.train.val_log_step is not None:
        pipeline = (
            (train_pipeline, val_pipeline) +
            # Cast(gt_labels, dtype=np.int16) +
            gp.TrainValProvider(step=config.train.val_log_step,
                                init_step=trained_until))
    else:
        pipeline = train_pipeline


    inputs={
        # 'anchor': anchor,
        'raw': raw,
        # 'gt_cell_indicator': cell_indicator,
        # 'gt_cell_center': cell_center,
    }
    outputs={
        0: pred_cell_indicator,
        1: maxima,
        2: raw_cropped,
    }
    if not config.model.train_only_cell_indicator:
        inputs['cell_mask'] = cell_mask
        inputs['gt_parent_vectors'] = parent_vectors
        outputs[3] = pred_parent_vectors
    # outputs={
    #     net_names['parent_vectors']: pred_parent_vectors,
    #     net_names['cell_indicator']: pred_cell_indicator,
    #     net_names['raw_cropped']: raw_cropped,
    #     net_names['maxima']: maxima,
    # }
    if config.train.cell_density:
        outputs[4] = pred_cell_density
        inputs['gt_cell_density'] = cell_density


    loss_inputs={
        'gt_cell_indicator': cell_indicator,
        'cell_indicator': pred_cell_indicator,
        'maxima': maxima,
        'gt_cell_center': cell_center,

    }
    if not config.model.train_only_cell_indicator:
        loss_inputs['cell_mask'] = cell_mask
        loss_inputs['gt_parent_vectors'] = parent_vectors
        loss_inputs['parent_vectors'] = pred_parent_vectors

    gradients={
        # net_names['parent_vectors']: grad_parent_vectors,
        # net_names['cell_indicator']: grad_cell_indicator,
    }

    snapshot_datasets = {
        raw: 'volumes/raw',
        # raw_tmp: 'volumes/raw_tmp',
        anchor: 'volumes/anchor',
        raw_cropped: 'volumes/raw_cropped',
        cell_indicator: 'volumes/cell_indicator',
        cell_center: 'volumes/cell_center',

        pred_cell_indicator: 'volumes/pred_cell_indicator',
        maxima: 'volumes/maxima',
        # grad_cell_indicator: 'volumes/grad_cell_indicator',
    }
    if not config.model.train_only_cell_indicator:
        snapshot_datasets[cell_mask] = 'volumes/cell_mask'
        snapshot_datasets[parent_vectors] = 'volumes/parent_vectors'
        snapshot_datasets[pred_parent_vectors] = 'volumes/pred_parent_vectors'
        # grad_parent_vectors: 'volumes/grad_parent_vectors',
    if config.train.cell_density:
        snapshot_datasets[cell_density] = 'volumes/cell_density'
        snapshot_datasets[pred_cell_density] = 'volumes/pred_cell_density'

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    opt = getattr(torch.optim, config.optimizerTorch.optimizer)(
        model.parameters(), **config.optimizerTorch.get_kwargs())

    data_to_save ={
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }
    torch.save(
        data_to_save,
        "train_net_checkpoint_0"
    )


    loss = LossWrapper(config, current_step=trained_until)

    pipeline = (
        pipeline +
        gp.torch.Train(
            model=model,
            loss=loss,
            optimizer=opt,
            checkpoint_basename=os.path.join(config.general.setup_dir, 'train_net'),
            inputs=inputs,
            outputs=outputs,
            loss_inputs=loss_inputs,
            gradients=gradients,
            log_dir=os.path.join(config.general.setup_dir, "train"),
            val_log_step=config.train.val_log_step,
            use_auto_mixed_precision=config.train.use_auto_mixed_precision,
            use_swa=config.train.use_swa,
            swa_every_it=config.train.swa_every_it,
            swa_start_it=config.train.swa_start_it,
            swa_freq_it=config.train.swa_freq_it,
            use_grad_norm=config.train.use_grad_norm,
            save_every=config.train.checkpoint_stride) +

        # visualize
        # gp.IntensityScaleShift(raw_cropped, scale=100.0, shift=0) +
        gp.Snapshot(snapshot_datasets,
            output_dir=os.path.join(config.general.setup_dir, 'snapshots'),
            output_filename='snapshot_{iteration}.hdf',
            additional_request=snapshot_request,
            every=config.train.snapshot_stride,
            dataset_dtypes={
                maxima: np.float32
            }) +
        gp.PrintProfilingStats(every=config.train.profiling_stride)
    )

    with gp.build(pipeline):

        logger.info("Starting training...")
        for i in range(trained_until, config.train.max_iterations):
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            logger.info(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)


def get_sources(config, raw, anchor, tracks, center_tracks, data_sources,
                cell_density=None, val=False):
    sources = []
    for ds in data_sources:
        d = ds.datafile.filename
        # if "/fast/AG_Kainmueller" in d:
        #     d = d.replace("/fast/AG_Kainmueller/phirsch/tracking/data_ny",
        #                   "/tmp")
        voxel_size = gp.Coordinate(ds.voxel_size)
        if not os.path.isdir(d):
            logger.info("trimming path %s", d)
            d = os.path.dirname(d)
        logger.info("loading data %s (val: %s)", d, val)
        data_config = load_config(os.path.join(d, "data_config.toml"))
        filename_zarr = os.path.join(
            d, data_config['general']['zarr_file'])
        filename_tracks = os.path.join(
            d, data_config['general']['tracks_file'])
        filename_daughters = os.path.join(
            d, data_config['general']['daughter_cells_file'])
        logger.info("creating source: %s (%s, %s, %s), divisions?: %s",
                    filename_zarr, ds.datafile.group,
                    filename_tracks, filename_daughters,
                    config.train.augment.divisions)
        file_size = data_config['general']['shape']
        limit_to_roi = gp.Roi(offset=ds.roi.offset, shape=ds.roi.shape)
        logger.info("limiting to roi: %s", limit_to_roi)
        datasets = {
            raw: ds.datafile.group,
            anchor: ds.datafile.group
        }
        array_specs = {
            raw: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size),
            anchor: gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)
        }
        if config.train.cell_density is not None:
            datasets[cell_density] = 'volumes/cell_density'
            array_specs[cell_density] = gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size)
        file_source = gp.ZarrSource(
            filename_zarr,
            datasets=datasets,
            nested="nested" in ds.datafile.group,
            # limit_to_roi=limit_to_roi,
            array_specs=array_specs) + \
            gp.Crop(raw, limit_to_roi)


        if config.train.augment.normalization is None or \
           config.train.augment.normalization == 'default':
            logger.info("default normalization")
            file_source = file_source + \
                gp.Normalize(raw,
                             factor=1.0/np.iinfo(data_config['stats']['dtype']).max)
        elif config.train.augment.normalization == 'minmax':
            mn = config.train.augment.norm_bounds[0]
            mx = config.train.augment.norm_bounds[1]
            logger.info("minmax normalization %s %s", mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn/2, mx=mx*2) + \
                NormalizeMinMax(raw, mn=mn, mx=mx, interpolatable=False)
            # file_source = file_source + Cast(raw, dtype=np.float32)
        elif config.train.augment.normalization == 'percminmax':
            mn = data_config['stats'][config.train.augment.perc_min]
            mx = data_config['stats'][config.train.augment.perc_max]
            logger.info("perc minmax normalization %s %s", mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn/2, mx=mx*2) + \
                NormalizeMinMax(raw, mn=mn, mx=mx)
        elif config.train.augment.normalization == 'mean':
            mean = data_config['stats']['mean']
            std = data_config['stats']['std']
            mn = data_config['stats'][config.train.augment.perc_min]
            mx = data_config['stats'][config.train.augment.perc_max]
            logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn, mx=mx) + \
                NormalizeMeanStd(raw, mean=mean, std=std)
        elif config.train.augment.normalization == 'median':
            median = data_config['stats']['median']
            mad = data_config['stats']['mad']
            mn = data_config['stats'][config.train.augment.perc_min]
            mx = data_config['stats'][config.train.augment.perc_max]
            logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn, mx=mx) + \
                NormalizeMedianMad(raw, median=median, mad=mad)
        else:
            raise RuntimeError("invalid normalization method %s",
                               config.train.augment.normalization)

        file_source = file_source + \
            gp.Pad(raw, None)#, value=config.train.augment.norm_bounds[0])

        file_resolution = gp.Coordinate(data_config['general']['resolution'])
        # file_resolution = gp.Coordinate([1, 11, 1, 1])
        scale = np.array(voxel_size)/np.array(file_resolution)
        logger.info("scaling tracks by %s", scale)
        track_source = (merge_sources(
            file_source,
            tracks,
            center_tracks,
            filename_tracks,
            filename_tracks,
            limit_to_roi,
            scale=scale,
            use_radius=config.train.use_radius) +
                        gp.RandomLocation(
                            ensure_nonempty=center_tracks,
                            p_nonempty=config.train.augment.reject_empty_prob,
                            point_balance_radius=config.train.augment.point_balance_radius,
                            pref="nodivtrain" if not val else "nodivval")
                        )
        if config.train.augment.divisions:
            div_source = (merge_sources(
                file_source,
                tracks,
                center_tracks,
                filename_tracks,
                filename_daughters,
                limit_to_roi,
                scale=scale,
                use_radius=config.train.use_radius) +
                          gp.RandomLocation(
                              ensure_nonempty=center_tracks,
                              p_nonempty=config.train.augment.reject_empty_prob,
                              point_balance_radius=config.train.augment.point_balance_radius,
                              pref="divtrain" if not val else "divval")
                          )

            track_source = (track_source, div_source) + \
                gp.RandomProvider(probabilities=[0.75, 0.25])
            # gp.Pad(raw, gp.Coordinate((0, 80, 55, 55))) + \

        # pad = np.array((input_size[0], 140, 100, 100))# * np.array(voxel_size)
        # source = (file_source, track_source) + \
        #     gp.MergeProvider()
        source = track_source
                  # doesn't work, crops/pads to late, has to be before random_location \
                  # gp.Crop(raw, limit_to_roi) + \
                  # gp.Pad(raw, None)

                        # gp.Coordinate(pad))

        if cell_density is not None:
            source = source + \
                gp.Crop(cell_density, limit_to_roi) + \
                gp.Pad(cell_density, None)
                       # gp.Coordinate(pad))


        sources.append(source)

    return sources


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
            logging.FileHandler(os.path.join(config.general.setup_dir, 'run.log'),
                                mode='a'),
            logging.StreamHandler(sys.stdout),
            # logging.StreamHandler(sys.stderr)
        ],
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')

    train_until(config)
