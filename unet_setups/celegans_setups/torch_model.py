import math

import torch

from funlib.learn.torch.models import UNet, ConvPass


def crop(x, shape):
    '''Center-crop x to match spatial dimensions given by shape.'''

    dims = len(x.size()) - len(shape)
    x_target_size = x.size()[:dims] + shape

    offset = tuple(
        (a - b)//2
        for a, b in zip(x.size(), x_target_size))

    slices = tuple(
        slice(o, o + s)
        for o, s in zip(offset, x_target_size))

    # print(x.size(), shape, slices)
    return x[slices]


def crop_to_factor(x, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.

    The crop could be done after the convolutions, but it is more efficient
    to do that before (feature maps will be smaller).
    '''

    shape = x.size()
    dims = len(x.size()) - 2
    spatial_shape = shape[-dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = tuple(
        sum(ks[d] - 1 for ks in kernel_sizes)
        for d in range(dims)
    )

    # we need (spatial_shape - convolution_crop) to be a multiple of
    # factor, i.e.:
    #
    # (s - c) = n*k
    #
    # we want to find the largest n for which s' = n*k + c <= s
    #
    # n = floor((s - c)/k)
    #
    # this gives us the target shape s'
    #
    # s' = n*k + c

    ns = (
        int(math.floor(float(s - c)/f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n*f + c
        for n, c, f in zip(ns, convolution_crop, factor)
    )

    if target_spatial_shape != spatial_shape:

        assert all((
            (t > c) for t, c in zip(
                target_spatial_shape,
                convolution_crop))
                   ), \
                   "Feature map with shape %s is too small to ensure " \
                   "translation equivariance with factor %s and following " \
                   "convolutions %s" % (
                       shape,
                       factor,
                       kernel_sizes)

        return crop(x, target_spatial_shape)

    return x


class UnetModelWrapper(torch.nn.Module):
    def __init__(self, config, current_step=0):
        super().__init__()
        self.config = config
        self.current_step = current_step

        num_fmaps = (config.model.num_fmaps
                     if isinstance(config.model.num_fmaps, list)
                     else [config.model.num_fmaps, config.model.num_fmaps])

        if config.model.unet_style == "split" or \
           config.model.unet_style == "single" or \
           self.config.model.train_only_cell_indicator:
            num_heads = 1
        elif config.model.unet_style == "multihead":
            num_heads = 2
        else:
            raise RuntimeError("invalid unet style, should be split, single or multihead")

        self.unet_cell_ind = UNet(
            in_channels=1,
            num_fmaps=num_fmaps[0],
            fmap_inc_factor=config.model.fmap_inc_factors,
            downsample_factors=config.model.downsample_factors,
            kernel_size_down=config.model.kernel_size_down,
            kernel_size_up=config.model.kernel_size_up,
            constant_upsample=config.model.constant_upsample,
            upsampling=config.model.upsampling,
            num_heads=num_heads,
        )

        if config.model.unet_style == "split" and \
           not self.config.model.train_only_cell_indicator:
            self.unet_par_vec = UNet(
                in_channels=1,
                num_fmaps=num_fmaps[1],
                fmap_inc_factor=config.model.fmap_inc_factors,
                downsample_factors=config.model.downsample_factors,
                kernel_size_down=config.model.kernel_size_down,
                kernel_size_up=config.model.kernel_size_up,
                constant_upsample=config.model.constant_upsample,
                upsampling=config.model.upsampling,
                num_heads=1,
            )

        self.cell_indicator_batched = ConvPass(num_fmaps[0], 1, [[1, 1, 1]],
                                               activation='Sigmoid')
        self.parent_vectors_batched = ConvPass(num_fmaps[1], 3, [[1, 1, 1]],
                                               activation=None)

        self.nms = torch.nn.MaxPool3d(config.model.nms_window_shape, stride=1, padding=0)

    def forward(self, raw):
        if self.config.model.latent_temp_conv:
            raw = torch.reshape(raw, [raw.size()[0], 1] + list(raw.size())[1:])
        else:
            raw = torch.reshape(raw, [1, 1] + list(raw.size()))
        model_out_1 = self.unet_cell_ind(raw)
        if self.config.model.unet_style != "multihead" or \
           self.config.model.train_only_cell_indicator:
            model_out_1 = [model_out_1]
        cell_indicator_batched = self.cell_indicator_batched(model_out_1[0])
        output_shape_1 = list(cell_indicator_batched.size())[1:]
        cell_indicator = torch.reshape(cell_indicator_batched, output_shape_1)

        if self.config.model.unet_style == "single":
            parent_vectors_batched = self.parent_vectors_batched(model_out_1[0])
            parent_vectors = torch.reshape(parent_vectors_batched, [3] + output_shape_1)
        elif self.config.model.unet_style == "split" and \
           not self.config.model.train_only_cell_indicator:
            model_par_vec = self.unet_par_vec(raw)
            parent_vectors_batched = self.parent_vectors_batched(model_par_vec)
            parent_vectors = torch.reshape(parent_vectors_batched, [3] + output_shape_1)
        else: # self.config.model.unet_style == "multihead"
            parent_vectors_batched = self.parent_vectors_batched(model_out_1[1])
            parent_vectors = torch.reshape(parent_vectors_batched, [3] + output_shape_1)

        maxima = self.nms(cell_indicator_batched)
        if not self.training:
            factor_product = None
            for factor in self.config.model.downsample_factors:
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f*ff
                        for f, ff in zip(factor, factor_product))
            maxima = crop_to_factor(
                maxima,
                factor=factor_product,
                kernel_sizes=[[1, 1, 1]])

        output_shape_2 = tuple(list(maxima.size())[1:])
        maxima = torch.reshape(maxima, output_shape_2)

        if not self.training:
            cell_indicator = crop(cell_indicator, output_shape_2)
            maxima = torch.eq(maxima, cell_indicator)
            if not self.config.model.train_only_cell_indicator:
                parent_vectors = crop(parent_vectors, output_shape_2)
        else:
            cell_indicator_cropped = crop(cell_indicator, output_shape_2)
            maxima = torch.eq(maxima, cell_indicator_cropped)

        if self.config.model.latent_temp_conv:
            raw_cropped = crop(raw[raw.size()[0]//2], output_shape_2)
        else:
            raw_cropped = crop(raw, output_shape_2)
        raw_cropped = torch.reshape(raw_cropped, output_shape_2)
        # print(torch.min(raw), torch.max(raw))
        # print(torch.min(raw_cropped), torch.max(raw_cropped))

        if self.config.model.train_only_cell_indicator:
            return cell_indicator, maxima, raw_cropped
        else:
            return cell_indicator, maxima, raw_cropped, parent_vectors
