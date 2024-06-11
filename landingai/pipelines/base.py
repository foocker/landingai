import torch
import numpy as np
from typing import Mapping


def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        # add compatibility for img_metas for mmlab models
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    elif get_class_name(data) == 'InputFeatures':
        # modelscope.preprocessors.nlp.InputFeatures
        return data
    elif get_class_name(data) == 'DataContainer':
        # mmcv.parallel.DataContainer
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')