from __future__ import print_function
import logging
import os
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("once", category=FutureWarning)

import numpy as np

from linajea.gunpowder import (TracksSource, AddParentVectors,
                               ShiftAugment, Clip, NoOp,
                               NormalizeMinMax, NormalizeMeanStd,
                               NormalizeMedianMad)
from linajea import load_config
import gunpowder as gp

# from utils import Cast
from base_gp_dataloader import BaseGpDataLoader

logger = logging.getLogger(__name__)


class GpDataLoader(BaseGpDataLoader):
    def __init__(self, config, *, mode, arrays=None, points=None):
        "docstring"
        super().__init__(config, mode=mode, arrays=arrays, points=points)

        if mode == "train":
            self.data_sources = config.train_data.data_sources
        elif mode == "test":
            self.data_sources = config.test_data.data_sources
        elif mode == "val":
            self.data_sources = config.validate_data.data_sources
        else:
            raise RuntimeError("invalid mode: %s", mode)

        # self.graph_keys["TRACKS"] = gp.GraphKey("TRACKS")
        # self.graph_keys["CENTER_TRACKS"] = gp.GraphKey("CENTER_TRACKS")

        self.voxel_size = gp.Coordinate(self.data_sources[0].voxel_size)

    def add_aug_ops(self):
        raw = self.array_keys["RAW"]
        augment = self.config.train.augment

        return (
            (gp.ElasticAugment(
                augment.elastic.control_point_spacing,
                augment.elastic.jitter_sigma,
                [augment.elastic.rotation_min*np.pi/180.0,
                 augment.elastic.rotation_max*np.pi/180.0],
                rotation_3d=augment.elastic.rotation_3d,
                subsample=augment.elastic.subsample,
                use_fast_points_transform=augment.elastic.use_fast_points_transform,
                spatial_dims=3,
                temporal_dim=True)
             if augment.elastic is not None else NoOp()) +

            (ShiftAugment(
                prob_slip=augment.shift.prob_slip,
                prob_shift=augment.shift.prob_shift,
                sigma=augment.shift.sigma,
                shift_axis=0)
             if augment.shift is not None else NoOp()) +

            (gp.SimpleAugment(
                mirror_only=augment.simple.mirror,
                transpose_only=augment.simple.transpose)
             if augment.simple is not None else NoOp()) +

            (gp.ZoomAugment(
                factor_min=augment.zoom.factor_min,
                factor_max=augment.zoom.factor_max,
                spatial_dims=augment.zoom.spatial_dims,
                order={raw: 1,
                       })
             if augment.zoom is not None else NoOp()) +


            (gp.NoiseAugment(
                raw,
                mode='gaussian',
                var=augment.noise_gaussian.var,
                clip=False,
                check_val_range=False)
             if augment.noise_gaussian is not None else NoOp()) +

            (gp.NoiseAugment(
                raw,
                mode='speckle',
                var=augment.noise_speckle.var,
                clip=False,
                check_val_range=False)
             if augment.noise_speckle is not None else NoOp()) +

            (gp.NoiseAugment(
                raw,
                mode='s&p',
                amount=augment.noise_saltpepper.amount,
                clip=False,
                check_val_range=False)
             if augment.noise_saltpepper is not None else NoOp()) +

            (gp.HistogramAugment(
                raw,
                # raw_tmp,
                range_low=augment.histogram.range_low,
                range_high=augment.histogram.range_high,
                z_section_wise=False)
            if augment.histogram is not None else NoOp())  +

            (gp.IntensityAugment(
                raw,
                scale_min=augment.intensity.scale[0],
                scale_max=augment.intensity.scale[1],
                shift_min=augment.intensity.shift[0],
                shift_max=augment.intensity.shift[1],
                z_section_wise=False,
                clip=False)
             if augment.intensity is not None else NoOp())
        )

    def add_post_aug_ops(self):
        tracks = self.graph_keys["TRACKS"]
        center_tracks = self.graph_keys["CENTER_TRACKS"]
        gt_cell_indicator = self.array_keys["GT_CELL_INDICATOR"]
        gt_parent_vectors = self.array_keys["GT_PARENT_VECTORS"]
        cell_mask = self.array_keys["CELL_MASK"]
        gt_cell_center = self.array_keys["GT_CELL_CENTER"]

        return (
            (AddParentVectors(
                tracks,
                gt_parent_vectors,
                cell_mask,
                array_spec=gp.ArraySpec(voxel_size=self.voxel_size),
                radius=self.config.train.parent_radius,
                move_radius=self.config.train.move_radius,
                dense=not self.config.general.sparse)
             if not self.config.model.train_only_cell_indicator else NoOp()) +

            (gp.Reject(
                mask=cell_mask,
                min_masked=0.0001,
                # reject_probability=augment.reject_empty_prob
            )
             if self.config.general.sparse else NoOp()) +

            (gp.Reject(
                # already done by random_location ensure_nonempty arg
                # no, not done, elastic augment changes roi, point might
                # be in outer part that gets cropped
                ensure_nonempty=center_tracks,
                # always reject emtpy batches in validation branch
                # reject_probability=augment.reject_empty_prob
            ) if self.mode == "val" else NoOp()) +

            gp.RasterizeGraph(
                tracks,
                gt_cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=self.voxel_size),
                settings=gp.RasterizationSettings(
                    radius=self.config.train.rasterize_radius,
                    mode='peak')) +

            gp.RasterizeGraph(
                tracks,
                gt_cell_center,
                array_spec=gp.ArraySpec(voxel_size=self.voxel_size),
                settings=gp.RasterizationSettings(
                    radius=(0.1,) + self.voxel_size[1:],
                    mode='point')) +

            (gp.PreCache(
                cache_size=self.config.train.cache_size,
                num_workers=self.config.train.job.num_workers)
             if self.config.train.job.num_workers > 1 else NoOp()) +

            gp.PrintProfilingStats(every=self.config.train.profiling_stride)
        )

    def normalize(self, file_source, data_config, raw):
        if self.config.train.augment.normalization is None or \
           self.config.train.augment.normalization == 'default':
            logger.info("default normalization")
            file_source = file_source + \
                gp.Normalize(raw,
                             factor=1.0/np.iinfo(data_config['stats']['dtype']).max)
        elif self.config.train.augment.normalization == 'minmax':
            mn = self.config.train.augment.norm_bounds[0]
            mx = self.config.train.augment.norm_bounds[1]
            logger.info("minmax normalization %s %s", mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn/2, mx=mx*2) + \
                NormalizeMinMax(raw, mn=mn, mx=mx, interpolatable=False)
            # file_source = file_source + Cast(raw, dtype=np.float32)
        elif self.config.train.augment.normalization == 'percminmax':
            mn = data_config['stats'][self.config.train.augment.perc_min]
            mx = data_config['stats'][self.config.train.augment.perc_max]
            logger.info("perc minmax normalization %s %s", mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn/2, mx=mx*2) + \
                NormalizeMinMax(raw, mn=mn, mx=mx)
        elif self.config.train.augment.normalization == 'mean':
            mean = data_config['stats']['mean']
            std = data_config['stats']['std']
            mn = data_config['stats'][self.config.train.augment.perc_min]
            mx = data_config['stats'][self.config.train.augment.perc_max]
            logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn, mx=mx) + \
                NormalizeMeanStd(raw, mean=mean, std=std)
        elif self.config.train.augment.normalization == 'median':
            median = data_config['stats']['median']
            mad = data_config['stats']['mad']
            mn = data_config['stats'][self.config.train.augment.perc_min]
            mx = data_config['stats'][self.config.train.augment.perc_max]
            logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
            file_source = file_source + \
                Clip(raw, mn=mn, mx=mx) + \
                NormalizeMedianMad(raw, median=median, mad=mad)
        else:
            raise RuntimeError("invalid normalization method %s",
                               self.config.train.augment.normalization)

        return file_source

    def add_inputs(self):
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

        raw = self.array_keys["RAW"]
        anchor = self.array_keys["ANCHOR"]
        tracks = self.graph_keys["TRACKS"]
        center_tracks = self.graph_keys["CENTER_TRACKS"]

        sources = []
        for ds in self.data_sources:
            d = ds.datafile.filename
            voxel_size = gp.Coordinate(ds.voxel_size)
            if not os.path.isdir(d):
                logger.info("trimming path %s", d)
                d = os.path.dirname(d)
            logger.info("loading data %s (mode: %s)", d, self.mode)
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
                        self.config.train.augment.divisions)
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
            file_source = gp.ZarrSource(
                filename_zarr,
                datasets=datasets,
                nested="nested" in ds.datafile.group,
                array_specs=array_specs) + \
                gp.Crop(raw, limit_to_roi)

            file_source = self.normalize(file_source, data_config, raw)

            file_source = file_source + \
                gp.Pad(raw, None)#, value=config.train.augment.norm_bounds[0])

            track_source = (
                merge_sources(
                    file_source,
                    tracks,
                    center_tracks,
                    filename_tracks,
                    filename_tracks,
                    limit_to_roi,
                    # scale=scale,
                    use_radius=self.config.train.use_radius) +
                gp.RandomLocation(
                    ensure_nonempty=center_tracks,
                    p_nonempty=self.config.train.augment.reject_empty_prob,
                    point_balance_radius=self.config.train.augment.point_balance_radius,
                    pref="nodiv" + self.mode)
            )
            if self.config.train.augment.divisions:
                div_source = (
                    merge_sources(
                        file_source,
                        tracks,
                        center_tracks,
                        filename_tracks,
                        filename_daughters,
                        limit_to_roi,
                        use_radius=self.config.train.use_radius) +
                    gp.RandomLocation(
                        ensure_nonempty=center_tracks,
                        p_nonempty=self.config.train.augment.reject_empty_prob,
                        point_balance_radius=self.config.train.augment.point_balance_radius,
                        pref="div" + self.mode)
                )

                track_source = (track_source, div_source) + \
                    gp.RandomProvider(probabilities=[0.75, 0.25])

            source = track_source

            sources.append(source)

        self.inputs = (
            tuple(sources) +
            gp.RandomProvider()
        )

        return self.inputs
