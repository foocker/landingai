#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections.abc import Mapping
import numpy as np


def compare_arguments_nested(print_content,
                             arg1,
                             arg2,
                             rtol=1.e-3,
                             atol=1.e-8,
                             ignore_unknown_type=True):
    type1 = type(arg1)
    type2 = type(arg2)
    if type1.__name__ != type2.__name__:
        if print_content is not None:
            print(
                f'{print_content}, type not equal:{type1.__name__} and {type2.__name__}'
            )
        return False

    if arg1 is None:
        return True
    elif isinstance(arg1, (int, str, bool, np.bool_, np.integer, np.str_)):
        if arg1 != arg2:
            if print_content is not None:
                print(f'{print_content}, arg1:{arg1}, arg2:{arg2}')
            return False
        return True
    elif isinstance(arg1, (float, np.floating)):
        if not np.isclose(arg1, arg2, rtol=rtol, atol=atol, equal_nan=True):
            if print_content is not None:
                print(f'{print_content}, arg1:{arg1}, arg2:{arg2}')
            return False
        return True
    elif isinstance(arg1, (tuple, list)):
        if len(arg1) != len(arg2):
            if print_content is not None:
                print(
                    f'{print_content}, length is not equal:{len(arg1)}, {len(arg2)}'
                )
            return False
        if not all([
                compare_arguments_nested(
                    None, sub_arg1, sub_arg2, rtol=rtol, atol=atol)
                for sub_arg1, sub_arg2 in zip(arg1, arg2)
        ]):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    elif isinstance(arg1, Mapping):
        keys1 = arg1.keys()
        keys2 = arg2.keys()
        if len(keys1) != len(keys2):
            if print_content is not None:
                print(
                    f'{print_content}, key length is not equal:{len(keys1)}, {len(keys2)}'
                )
            return False
        if len(set(keys1) - set(keys2)) > 0:
            if print_content is not None:
                print(f'{print_content}, key diff:{set(keys1) - set(keys2)}')
            return False
        if not all([
                compare_arguments_nested(
                    None, arg1[key], arg2[key], rtol=rtol, atol=atol)
                for key in keys1
        ]):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    elif isinstance(arg1, np.ndarray):
        arg1 = np.where(np.equal(arg1, None), np.NaN, arg1).astype(dtype=float)
        arg2 = np.where(np.equal(arg2, None), np.NaN, arg2).astype(dtype=float)
        if not all(
                np.isclose(arg1, arg2, rtol=rtol, atol=atol,
                           equal_nan=True).flatten()):
            if print_content is not None:
                print(f'{print_content}')
            return False
        return True
    else:
        if ignore_unknown_type:
            return True
        else:
            raise ValueError(f'type not supported: {type1}')
