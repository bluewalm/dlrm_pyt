# Copyright (c) 2023 BLUEWALM. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import ctypes
import torch
from typing import Optional, List

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        if dtype == np.float16:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.int16))
            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            self._host.dtype = np.float16
        else:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='same_kind')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_tensor_names = [binding for binding in tensor_names if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT]
    output_tensor_names = [binding for binding in tensor_names if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT]
    # bind inputs
    for binding in input_tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        if engine.has_implicit_batch_dimension:
            size *= engine.max_batch_size
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)
        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))
        # Append to the appropriate list.
        inputs.append(bindingMemory)
    # bind outputs
    for binding in output_tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        bound_shape = engine.get_tensor_profile_shape(input_tensor_names[-1], profile_idx)[-1]
        shape = engine.get_tensor_shape(binding)
        shape = tuple(x if x >= 0 else bound_shape[i] for i,x in enumerate(shape))
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        if engine.has_implicit_batch_dimension:
            size *= engine.max_batch_size
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)
        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))
        # Append to the appropriate list.
        outputs.append(bindingMemory)
    # return result
    return inputs, outputs, bindings, stream


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    def execute_async():
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async)


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    def execute_async():
        context.execute_async_v2(bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async)


class Internal:
    def __init__(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            runtime.max_threads = 10
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, profile_idx=0)
        self.trt_context = self.engine.create_execution_context()
    
    def __call__(self, numerical_features, categorical_features):
        device = numerical_features.device
        numerical_features = numerical_features.cpu().numpy()
        categorical_features = categorical_features.cpu().numpy()
        # load input
        self.trt_context.set_input_shape('numerical_features', numerical_features.shape)
        self.trt_context.set_input_shape('categorical_features', categorical_features.shape)
        self.inputs[0].host = numerical_features
        self.inputs[1].host = categorical_features
        # do inference
        trt_outputs = do_inference_v2(self.trt_context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        output = trt_outputs[0].reshape(numerical_features.shape[0], -1)
        output = torch.from_numpy(output).to(device=device)
        return output
    
    def cleanup(self):
        free_buffers(self.inputs, self.outputs, self.stream)


class Model:
    def __init__(self, engine_file_path):
        if not os.path.isfile(engine_file_path):
            raise IOError("failed to load tensorrt engine")
        self.engine_file_path = engine_file_path
    
    def __enter__(self):
        self.internal = Internal(self.engine_file_path)
        return self.internal
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.internal.cleanup()

