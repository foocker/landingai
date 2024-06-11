#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import logging
import os
import sys
import pycuda.driver as cuda
import tensorrt as trt

from cuda import cudart
import  numpy as np

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print(f"WARNING: {data_path} does not exist. Trying {data_dir} instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)

def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}\n{:}".format(filename, data_paths, err_msg))
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def allocate_buffers_v3(engine, input_data:dict = None):
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    
    bufferH = [np.ascontiguousarray(data) for data in input_data.values()]  # TODO 传入numpy格式
    for i in range(nInput, nIO):
        # 这里动态输出，存在shape有-1情况。 TODO 
        bufferH.append(np.empty(engine.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        
    return nIO, nInput, lTensorName, bufferH, bufferD

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def do_inference_v3(nIO, nInput,lTensorName, context, bufferH, bufferD, stream=0):
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(stream)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
    for b in bufferD:
        cudart.cudaFree(b)
        
    # for i in range(nIO):
    #     print(lTensorName[i])
    #     print(bufferH[i])
    
    log.info("Succeeded running model in TensorRT!")

    return bufferH, lTensorName

def infer_v3(trtFile, mode='fp16'):
    import torch, time
    datatype = torch.float16 if mode=='fp16' else torch.int8
    if mode == "fp16":
        source = torch.randn(1, 3, 256, 256, dtype=datatype)
        driving_value = torch.randn(1, 15, 3, dtype=datatype)
        source_value = torch.randn(1, 15, 3, dtype=datatype)
    else:
        min_value = -128
        max_value = 127
        source = torch.randint(min_value, max_value + 1, size=(1, 3, 256, 256), dtype=datatype)
        driving_value = torch.randint(min_value, max_value + 1, size=(1, 15, 3), dtype=datatype)
        source_value = torch.randint(min_value, max_value + 1, size=(1, 15, 3), dtype=datatype)
    input_data = {"source":source.cpu().numpy(), "driving_value":driving_value.cpu().numpy(), "source_value":source_value.cpu().numpy()}

    # logger = trt.Logger(trt.Logger.VERBOSE)
    # with open(trtFile, "rb") as f, trt.Runtime(logger) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
    with open(trtFile, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        st = time.time()
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

        shapes = [[1, 3, 256, 256], [1, 15, 3],[1, 15, 3]]
        for i, ina in enumerate(lTensorName):
            context.set_input_shape(ina, shapes[i])
        for i in range(nIO):
            print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

        bufferH = []
        for k, v in input_data.items():
            bufferH.append(np.ascontiguousarray(v))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

        context.execute_async_v3(0)

        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        for i in range(nIO):
            print(lTensorName[i])
            print(bufferH[i])

        for b in bufferD:
            cudart.cudaFree(b)

        print("Succeeded running model in TensorRT!")
        cost = time.time()
        print(f'cost time of {mode} is :', cost - st)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, bUseFP16Mode=True, dynamic=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_vsersion = int(trt.__version__.split('.')[1])
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB
        
        if bUseFP16Mode:
            self.config.set_flag(trt.BuilderFlag.FP16)
        
        # if bUseINT8Mode and calibrator is not None:
        #     self.config.set_flag(trt.BuilderFlag.INT8)
        #     self.config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)
        
        self.profile = self.builder.create_optimization_profile()
        
        self.batch_size = None
        self.network = None
        self.parser = None
        self.dynamic = dynamic
        
        cudart.cudaDeviceSynchronize()
        
    def prepare_data(self, **kwargs):
        pass

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error(f"Failed to load ONNX file: {onnx_path}")
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)
                
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}"
            )
        for output in outputs:
            log.info(
                f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}"
            )
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=25000,
                      calib_batch_size=8, calib_preprocessor=None, dynamics_shapes=None):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        :dynamics_shapes: {"name1":([1,3,h1,w1], [4,3,h2,w2], [8, 3, h3, w3]), "name2":([1,3,h1,w1], [4,3,h2,w2], [8, 3, h3, w3])}
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info(f"Building {precision} Engine in {engine_path}")

        # inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
            
        if self.dynamic:
            for k, v in dynamics_shapes.items():
                self.profile.set_shape(k, v[0], v[1], v[2])  # len(v)==3 TODO 
            self.config.add_optimization_profile(self.profile)
                
        if self.trt_vsersion == 5:
            with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
                log.info("Serializing engine to file: {:}".format(engine_path))
                f.write(engine.serialize())
        elif self.trt_vsersion == 6:
            engineString = self.builder.build_serialized_network(self.network, self.config)
            if engineString == None:
                log.info("Failed building engine!")
                exit()
            log.info("Succeeded building engine!")
            with open(engine_path, "wb") as f:
                f.write(engineString)
        else:
            pass
    
    def load_engine(self, engineString):
        pass
