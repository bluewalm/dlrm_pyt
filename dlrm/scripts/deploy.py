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


import os
import sys
import json
import torch
import tqdm
import math
import time
import copy
import itertools
import subprocess
import numpy as np
from argparse import ArgumentParser

from dlrm.data.feature_spec import FeatureSpec
from dlrm.model.distributed import DistributedDlrm
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed

import datetime
from time import time
from absl import app, flags

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes

import tensorrt as trt
import dlrm.scripts.tensorrt_lib as tensorrt_lib
import pytorch_embeddings
from pytorch_embeddings import decomposed_embeddings


package_path = os.path.dirname(os.path.realpath(pytorch_embeddings.__file__))
plugin_path = os.path.join(package_path, 'frobenius_operator', 'tensorrt', 'libfrobenius_operatorPlugin.so')


FLAGS = flags.FLAGS


# Basic run settings
flags.DEFINE_integer("seed", 1, "Random seed")

# Training flags
flags.DEFINE_integer("batch_size", 65536, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 65536, "Batch size used for testing/validation")

# Model configuration
flags.DEFINE_enum("embedding_type", "multi_table",
                  ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"],
                  help="The type of the embedding operation to use")
flags.DEFINE_list("embedding_compression_type", None,
               help="The type of the embedding compression to use. To be given per embedding table. \
                    Can be either 'native' or 'frobenius'. \
                    Only relevant when 'embedding_type' is set to 'multi_table'. Otherwise ignored. ")
flags.DEFINE_list("frobenius_rank", None, "the dimension of factor-[l,h] we reduce over. given per embedding table. ")
flags.DEFINE_list("frobenius_blocks", None, "given per frobenius embedding table. ")
flags.DEFINE_boolean("alternating_gradients", False, "alternate gradients for frobenius training")
flags.DEFINE_boolean("silent", False, "do not display verbose information about decomposed embeddings")
flags.DEFINE_boolean("onchip_memory", True, "memory reduction for frobenius embeddings")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")
flags.DEFINE_enum("interaction_op", default="dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

# Data configuration
flags.DEFINE_string("dataset", None, "Path to dataset directory")
flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                    help="Name of the feature spec file in the dataset directory")
flags.DEFINE_enum("dataset_type", default="parametric", enum_values=['synthetic_gpu', 'parametric'],
                  help='The type of the dataset to use')
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, "
                     "by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` "
                     "to ensure that the indices match table sizes")

# Synthetic data configuration
flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2 ** 15 * 1024),
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinalities of variables to use with the synthetic dataset.")
flags.DEFINE_integer("synthetic_dataset_numerical_features", default='13',
                     help="Number of numerical features to use with the synthetic dataset")
flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                     help="Create a temporary synthetic dataset based on a real one. "
                          "Uses --dataset and --feature_spec"
                          "Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features."
                          "--synthetic_dataset_num_entries is still required")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")

# Saving and logging flags
flags.DEFINE_integer("benchmark_warmup_steps", 10,
                     "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("amp", False, "enable fp16 precision for inference")

# inference benchmark
flags.DEFINE_integer("inference_benchmark_steps", 10000,
                     "Number of steps for measuring inference latency and throughput")

# Miscellaneous
flags.DEFINE_boolean("optimized_mlp", False, "Use an optimized implementation of MLP from apex")
flags.DEFINE_enum("auc_device", default="GPU", enum_values=['GPU', 'CPU'],
                  help="Specifies where ROC AUC metric is calculated")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False,
                     "Sort features from the bottom model, useful when using saved "
                     "checkpoint in different device configurations")


def validate_flags(cat_feature_count):
    if FLAGS.max_table_size is not None and not FLAGS.hash_indices:
        raise ValueError('Hash indices must be True when setting a max_table_size')

    if FLAGS.base_device == 'cpu':
        if FLAGS.embedding_type in ('joint_fused', 'joint_sparse'):
            print('WARNING: CUDA joint embeddings are not supported on CPU')
            FLAGS.embedding_type = 'joint'

        if FLAGS.amp:
            print('WARNING: Automatic mixed precision not supported on CPU')
            FLAGS.amp = False

    if FLAGS.embedding_type == 'custom_cuda':
        if (not is_distributed()) and FLAGS.embedding_dim == 128 and cat_feature_count == 26:
            FLAGS.embedding_type = 'joint_fused'
        else:
            FLAGS.embedding_type = 'joint_sparse'

    if FLAGS.embedding_type == 'joint_fused' and FLAGS.embedding_dim != 128:
        print('WARNING: Joint fused can be used only with embedding_dim=128. Changed embedding type to joint_sparse.')
        FLAGS.embedding_type = 'joint_sparse'

    if FLAGS.dataset is None and (FLAGS.dataset_type != 'synthetic_gpu' or
                                  FLAGS.synthetic_dataset_use_feature_spec):
        raise ValueError('Dataset argument has to specify a path to the dataset')

    FLAGS.top_mlp_sizes = [int(x) for x in FLAGS.top_mlp_sizes]
    FLAGS.bottom_mlp_sizes = [int(x) for x in FLAGS.bottom_mlp_sizes]

    # TODO check that bottom_mlp ends in embedding_dim size


def load_feature_spec(flags):
    if flags.dataset_type == 'synthetic_gpu' and not flags.synthetic_dataset_use_feature_spec:
        num_numerical = flags.synthetic_dataset_numerical_features
        categorical_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        return FeatureSpec.get_default_feature_spec(number_of_numerical_features=num_numerical,
                                             categorical_feature_cardinalities=categorical_sizes)
    fspec_path = os.path.join(flags.dataset, flags.feature_spec)
    return FeatureSpec.from_yaml(fspec_path)


def dist_evaluate(model, data_loader):
    """Test distributed DLRM model

    Args:
        model (DistDLRM):
        data_loader (torch.utils.data.DataLoader):
    """
    device = FLAGS.base_device
    world_size = dist.get_world_size()

    batch_sizes_per_gpu = [FLAGS.test_batch_size // world_size for _ in range(world_size)]
    test_batch_size = sum(batch_sizes_per_gpu)

    if FLAGS.test_batch_size != test_batch_size:
        print(f"Rounded test_batch_size to {test_batch_size}")

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))

    with torch.no_grad():

        # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
        # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
        y_true = []
        y_score = []
        data_stream = torch.cuda.Stream()

        batch_iter = prefetcher(iter(data_loader), data_stream)

        for step in range(len(data_loader)):
            numerical_features, categorical_features, click = next(batch_iter)
            torch.cuda.synchronize()

            last_batch_size = None
            if click.shape[0] != test_batch_size:  # last batch
                last_batch_size = click.shape[0]
                padding_size = test_batch_size - last_batch_size

                if numerical_features is not None:
                    padding_numerical = torch.empty(
                        padding_size, numerical_features.shape[1],
                        device=numerical_features.device, dtype=numerical_features.dtype)
                    numerical_features = torch.cat((numerical_features, padding_numerical), dim=0)

                if categorical_features is not None:
                    padding_categorical = torch.ones(
                        padding_size, categorical_features.shape[1],
                        device=categorical_features.device, dtype=categorical_features.dtype)
                    categorical_features = torch.cat((categorical_features, padding_categorical), dim=0)

            with torch.cuda.amp.autocast(enabled=FLAGS.amp):
                output = model(numerical_features, categorical_features)
                output = output.squeeze()
                output = output.float()

            if world_size > 1:
                output_receive_buffer = torch.empty(test_batch_size, device=device)
                torch.distributed.all_gather(list(output_receive_buffer.split(batch_sizes_per_gpu)), output)
                output = output_receive_buffer

            if last_batch_size is not None:
                output = output[:last_batch_size]

            if FLAGS.auc_device == "CPU":
                click = click.cpu()
                output = output.cpu()

            y_true.append(click)
            y_score.append(output)

        if is_main_process():
            y_true = torch.cat(y_true)
            y_score = torch.sigmoid(torch.cat(y_score)).float()
            auc = utils.roc_auc_score(y_true, y_score)
        else:
            auc = None

        if world_size > 1:
            torch.distributed.barrier()

    return auc


def inference_benchmark(model, data_loader):
    model.eval()
    base_device = FLAGS.base_device
    latencies = []
    
    with torch.no_grad():
        for step, (numerical_features, categorical_features, _) in enumerate(data_loader):
            if step > FLAGS.inference_benchmark_steps:
                break
            
            numerical_features = numerical_features.to(base_device)
            if FLAGS.amp:
                numerical_features = numerical_features.half()
            
            categorical_features = categorical_features.to(device=base_device, dtype=torch.int64)
            
            torch.cuda.synchronize()
            step_start_time = time()
            inference_result = model(numerical_features, categorical_features).squeeze()
            torch.cuda.synchronize()
            step_time = time() - step_start_time
            
            if step >= FLAGS.benchmark_warmup_steps:
                latencies.append(step_time)
    
    return latencies


def execute(command):
    ''' 
        execute command; capture and print stdout
        return stdout 
    ''' 
    command = command.split()
    outputs = []
    stdout = subprocess.PIPE
    with subprocess.Popen(command, stdout=stdout, bufsize=1, 
                        universal_newlines=True) as process:
        for line in process.stdout:
            line = line[:-1]
            outputs.append(line)
            print(line)
    output = ''.join(outputs)
    return output


def export_inputs(sample):
    numerical_features = sample[0].to(device='cpu').numpy()
    numerical_features.tofile("numerical_features.dat")
    categorical_features = sample[1].to(device='cpu').numpy()
    categorical_features.tofile("categorical_features.dat")


def shape_to_str(shape):
    return "x".join([str(i) for i in shape])


def convert_to_tensorrt(amp, sample):
    command = "trtexec --onnx=./model.onnx --staticPlugins=" + str(plugin_path)
    command += " --loadInputs=numerical_features:./numerical_features.dat,categorical_features:./categorical_features.dat"
    if amp:
        command += " --inputIOFormats=fp16:chw,int32:chw"
        command += " --outputIOFormats=fp16:chw"
    else:
        command += " --inputIOFormats=fp32:chw,int32:chw"
        command += " --outputIOFormats=fp32:chw"
    if amp:
        command += " --fp16"
    numerical_features_shape = shape_to_str(sample[0].shape)
    categorical_features_shape = shape_to_str(sample[1].shape)
    min_shapes = "numerical_features:" + numerical_features_shape + ",categorical_features:" + categorical_features_shape
    command += " --minShapes=" + min_shapes
    opt_shapes = "numerical_features:" + numerical_features_shape + ",categorical_features:" + categorical_features_shape
    command += " --optShapes=" + opt_shapes
    max_shapes = "numerical_features:" + numerical_features_shape + ",categorical_features:" + categorical_features_shape
    command += " --maxShapes=" + max_shapes
    command += " --builderOptimizationLevel=5"
    command += " --maxAuxStreams=5"
    command += " --memPoolSize=workspace:16384"
    command += " --saveEngine=./model.engine"
    command += " --skipInference"
    execute(command)
    engine_file_path = "./model.engine"
    return engine_file_path


def main(argv):
    torch.manual_seed(FLAGS.seed)
    
    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed_mode(backend=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device
    
    feature_spec = load_feature_spec(FLAGS)
    
    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(cat_feature_count)
    
    FLAGS.set_default("test_batch_size", FLAGS.test_batch_size // world_size * world_size)
    
    feature_spec = load_feature_spec(FLAGS)
    world_embedding_sizes = get_embedding_sizes(feature_spec, max_table_size=FLAGS.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
    device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=world_size)
    
    batch_sizes_per_gpu = get_gpu_batch_sizes(FLAGS.batch_size, num_gpus=world_size)
    batch_indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))  # todo what does this do
    
    # Embedding sizes for each GPU
    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping['embedding'][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()
    
    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping['bottom_mlp'] else None
    
    _, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping, feature_spec=feature_spec)
    
    decomposed_embeddings_flags = {
        'embedding_type' : FLAGS.embedding_compression_type, 
        'frobenius_rank' : [int(x) for x in FLAGS.frobenius_rank] if FLAGS.frobenius_rank is not None else None, 
        'frobenius_blocks' : [int(x) for x in FLAGS.frobenius_blocks] if FLAGS.frobenius_blocks is not None else None, 
        'alternating_gradients' : FLAGS.alternating_gradients, 
        'silent' : FLAGS.silent, 
        'onchip_memory' : FLAGS.onchip_memory 
    }
    
    with decomposed_embeddings(**decomposed_embeddings_flags):
        model = DistributedDlrm(
            vectors_per_gpu=device_mapping['vectors_per_gpu'],
            embedding_device_mapping=device_mapping['embedding'],
            embedding_type=FLAGS.embedding_type,
            embedding_dim=FLAGS.embedding_dim,
            world_num_categorical_features=len(world_categorical_feature_sizes),
            categorical_feature_sizes=categorical_feature_sizes,
            num_numerical_features=num_numerical_features,
            hash_indices=FLAGS.hash_indices,
            bottom_mlp_sizes=bottom_mlp_sizes,
            top_mlp_sizes=FLAGS.top_mlp_sizes,
            interaction_op=FLAGS.interaction_op,
            fp16=FLAGS.amp,
            use_cpp_mlp=FLAGS.optimized_mlp,
            bottom_features_ordered=FLAGS.bottom_features_ordered,
            device=device
        )
    
    dist.setup_distributed_print(is_main_process())
    
    checkpoint_loader = make_distributed_checkpoint_loader(
        device_mapping=device_mapping, 
        rank=rank, 
        config=FLAGS.flag_values_dict()
    )
    
    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)
    
    # use pure fp16 for inference
    if FLAGS.amp:
        model = model.half()
    
    # switch to eval mode
    model.eval()
    
    print(model)
    
    auc_pyt = dist_evaluate(model, data_loader_test)
    
    # get samples
    numerical_features, categorical_features, _ = next(iter(data_loader_test))
    numerical_features = numerical_features.to(FLAGS.base_device)
    if FLAGS.amp:
        numerical_features = numerical_features.half()
    categorical_features = categorical_features.to(device=FLAGS.base_device, dtype=torch.int64)
    sample = (numerical_features, categorical_features)
    
    # ts export
    model_traced = torch.jit.trace(model, sample)
    # turn off torchscript recompilations
    with torch.jit.optimized_execution(False):
        # ts_eval
        auc_ts = dist_evaluate(model_traced, data_loader_test)
        # onnx_export
        torch.onnx.export(model_traced, sample, "model.onnx", verbose=False, 
                       opset_version=18, export_params=True, keep_initializers_as_inputs=False, 
                       custom_opsets={"trt.plugins" : 1}, do_constant_folding=True, 
                       input_names=['numerical_features', 'categorical_features'], output_names=['output'], 
                       dynamic_axes={'numerical_features' : {0 : 'batch_size'}, 'categorical_features' : {0 : 'batch_size'}, 
                                    'output' : {0 : 'batch_size'}})
    
    # trt export
    export_inputs(sample)
    engine_file_path = convert_to_tensorrt(FLAGS.amp, sample)
    
    # trt eval
    with tensorrt_lib.Model(engine_file_path) as model_trt:
        auc_trt = dist_evaluate(model_trt, data_loader_test)
    
    # print stats
    auc_data = {
            'pytorch (auc)'   : auc_pyt, 
            'torchscript (auc)' : auc_ts, 
            'tensorrt (auc)'   : auc_trt, 
    }
    print(json.dumps(auc_data, indent=4))
    
    # measure perf for pytorch model
    latencies = inference_benchmark(model, data_loader_test)
    
    # calculate throughput and latency data
    pytorch_perf_data = {
        'pytorch mean_throughput' : FLAGS.test_batch_size / np.mean(latencies), 
        'pytorch mean_latency' : np.mean(latencies), 
        'pytorch p90_latency' : np.percentile(latencies, 0.90), 
        'pytorch p95_latency' : np.percentile(latencies, 0.95),
        'pytorch p99_latency' : np.percentile(latencies, 0.99)
    }
    print(json.dumps(pytorch_perf_data, indent=4))
    
    with torch.jit.optimized_execution(False):
        # measure perf for torchscript model
        latencies = inference_benchmark(model_traced, data_loader_test)
    
    # calculate throughput and latency data
    torchscript_perf_data = {
        'torchscript mean_throughput' : FLAGS.test_batch_size / np.mean(latencies), 
        'torchscript mean_latency' : np.mean(latencies), 
        'torchscript p90_latency' : np.percentile(latencies, 0.90), 
        'torchscript p95_latency' : np.percentile(latencies, 0.95),
        'torchscript p99_latency' : np.percentile(latencies, 0.99)
    }
    print(json.dumps(torchscript_perf_data, indent=4))
    
    # measure trt latency
    command = "trtexec --loadEngine=./model.engine --staticPlugins=" + str(plugin_path)
    if FLAGS.amp:
        command += " --inputIOFormats=fp16:chw,int32:chw"
        command += " --outputIOFormats=fp16:chw"
    else:
        command += " --inputIOFormats=fp32:chw,int32:chw"
        command += " --outputIOFormats=fp32:chw"
    if FLAGS.amp:
        command += " --fp16"
    command += " --loadInputs=numerical_features:./numerical_features.dat,categorical_features:./categorical_features.dat"
    numerical_features_shape = shape_to_str(sample[0].shape)
    categorical_features_shape = shape_to_str(sample[1].shape)
    command += " --shapes=numerical_features:" + numerical_features_shape + ",categorical_features:" + categorical_features_shape
    command += " --iterations=" + str(FLAGS.inference_benchmark_steps + FLAGS.benchmark_warmup_steps)
    command += " --avgRuns=" + str(FLAGS.inference_benchmark_steps)
    command += " --infStreams=1"
    command += " --noDataTransfers"
    command += " --useSpinWait"
    trtexec_output = execute(command)
    index = trtexec_output.find('=== Performance summary ===')
    result = trtexec_output[index:]
    print(result)


if __name__ == '__main__':
    app.run(main)

