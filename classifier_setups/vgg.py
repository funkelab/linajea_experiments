from linajea import load_config
from funlib.learn.torch.models import Vgg3D


def get_cell_cycle_model(config_file):
    config = load_config(config_file)
    input_shape = config['input_shape']  # including channels
    channels = input_shape[0]
    input_size = input_shape[1:]
    num_classes = config['num_classes']
    downsample_factors = config['downsample_factors']
    fmap_inc_factors = config['fmap_inc_factors']
    fmaps = config['num_fmaps']
    model = Vgg3D(
            input_size,
            fmaps=fmaps,
            downsample_factors=downsample_factors,
            fmap_inc=fmap_inc_factors,
            n_convolutions=(2,)*len(downsample_factors),
            output_classes=num_classes,
            input_fmaps=channels)
    return model
