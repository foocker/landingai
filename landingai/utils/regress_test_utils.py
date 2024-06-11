# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

from .test_utils import compare_arguments_nested


def numpify_tensor_nested(tensors, reduction=None, clip_value=10000):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (Mapping, dict)):
        return OrderedDict({
            k: numpify_tensor_nested(t, reduction, clip_value)
            for k, t in tensors.items()
        })
    if isinstance(tensors, list):
        return list(
            numpify_tensor_nested(t, reduction, clip_value) for t in tensors)
    if isinstance(tensors, tuple):
        return tuple(
            numpify_tensor_nested(t, reduction, clip_value) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        t: np.ndarray = tensors.cpu().numpy()
        if clip_value is not None:
            t = np.where(t > clip_value, clip_value, t)
            t = np.where(t < -clip_value, -clip_value, t)
        if reduction == 'sum':
            return t.sum(dtype=float)
        elif reduction == 'mean':
            return t.mean(dtype=float)
        return t
    return tensors


def detach_tensor_nested(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (Mapping, dict)):
        return OrderedDict(
            {k: detach_tensor_nested(t)
             for k, t in tensors.items()})
    if isinstance(tensors, list):
        return list(detach_tensor_nested(t) for t in tensors)
    if isinstance(tensors, tuple):
        return tuple(detach_tensor_nested(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors

