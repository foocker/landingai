source = torch.randn(1, 3, 256, 256)
driving_value = torch.randn(1, 15, 3)
source_value = torch.randn(1, 15, 3)

SAMPLES  = [{"source":torch.randn(1, 3, 256, 256),
             "driving_value":torch.randn(1, 15, 3),
             "source_value":torch.randn(1, 15, 3)} for _ in range(32)]

print(SAMPLES[0]['source'].shape)

import trt_infer
from tqdm import tqdm
import tensorrt as trt

def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        for sample in tqdm(samples, desc='TensorRT is running...'):
            _extracted_from_infer_trt_13(sample, inputs)
            for _ in tqdm(samples, desc='TensorRT is running...'):
                trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)
    return results

def infer_trt_v3(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    results = []
    with engine.create_execution_context() as context:
        nIO, nInput, lTensorName, bufferH, bufferD = trt_infer.allocate_buffers_v3(context.engine)
        for sample in tqdm(samples, desc='TensorRT is running...'):
            # _extracted_from_infer_trt_13(sample, inputs)
            for _ in tqdm(samples, desc='TensorRT is running...'):
                bufferHResult, lTensorName = trt_infer.do_inference_v3(nIO, nInput, lTensorName, bufferH, bufferD, stream=0)
                results.append(bufferHResult)
    return results

from typing import List, Union
import numpy as np
import torch

def convert_any_to_numpy(
    x: Union[torch.Tensor, np.ndarray, int, float, list, tuple],
    accept_none: bool=True) -> np.ndarray:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, np.ndarray): return x
    elif isinstance(x, int) or isinstance(x, float): return np.array([x, ])
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none: return None
        if x.numel() == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if x.numel() >= 1: return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as numpy type.')
    
def _extracted_from_infer_trt_13(sample, inputs):
    inputs[0].host = convert_any_to_numpy(sample['source'])
    inputs[1].host = convert_any_to_numpy(sample['driving_value'])
    inputs[2].host = convert_any_to_numpy(sample['source_value'])