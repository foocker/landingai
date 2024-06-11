import gc
import os
import torch
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Union, Tuple, List, Type

# replace it by modelscope TODO 
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from landingai.triton.auto_config.configuration import EngineType, Configuration
from landingai.triton.auto_config.configuration_decoder import ConfigurationDec

from landingai.benchmarks.utils import track_infer_time, generate_multiple_inputs, print_timings, setup_logging

from landingai.backends.pytorch_utils import infer_classification_pytorch, infer_text_generation
from landingai.utils.args import parse_args
from landingai.utils.accuracy import check_accuracy


def launch_inference(
    infer: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    inputs: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    nb_measures: int,
) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[float]]:
    """
    Perform inference and measure latency.

    :param infer: a lambda which will perform the inference
    :param inputs: tensor compatible with the lambda (Torch tensor for Pytorch, or numpy otherwise)
    :param nb_measures: number of measures to perform for the latency measure
    :return: a tuple of model output and inference latencies
    """
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(batch_input)
        outputs.append(output)
    time_buffer: List[int] = list()
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(inputs[0])
    return outputs, time_buffer


def main(commands: argparse.Namespace):
    torch.cuda.empty_cache()
    setup_logging(level=logging.INFO if commands.verbose else logging.WARNING)
    logging.info("running with commands: %s", commands)
    # set seeds:
    torch.manual_seed(commands.seed)
    np.random.seed(commands.seed)
    torch.set_num_threads(commands.nb_threads)

    # set device
    if commands.device is None:
        commands.device = "cuda" if torch.cuda.is_available() else "cpu"

    if commands.device == "cpu" and "tensorrt" in commands.backend:
        raise Exception("can't perform inference on CPU and use Nvidia TensorRT as backend")

    if commands.task == "text-generation" and commands.generative_model == "t5" and "tensorrt" in commands.backend:
        raise Exception("TensorRT is not supported yet for T5 transformation")

    if len(commands.seq_len) == len(set(commands.seq_len)) and "tensorrt" in commands.backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    run_on_cuda: bool = commands.device.startswith("cuda")
    if run_on_cuda:
        assert torch.cuda.is_available(), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"

    # set authentication
    if isinstance(commands.auth_token, str) and commands.auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(commands.auth_token, str):
        auth_token = commands.auth_token
    else:
        auth_token = None

    Path(commands.output).mkdir(parents=True, exist_ok=True)
    tokenizer_path = commands.tokenizer if commands.tokenizer else commands.model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=commands.model, use_auth_token=auth_token
    )
    input_names: List[str] = tokenizer.model_input_names
    if commands.task == "classification":
        model_pytorch = AutoModelForSequenceClassification.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "token-classification":
        model_pytorch = AutoModelForTokenClassification.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "question-answering":
        model_pytorch = AutoModelForQuestionAnswering.from_pretrained(commands.model, use_auth_token=auth_token)
    elif commands.task == "text-generation" and commands.generative_model == "gpt":
        model_pytorch = AutoModelForCausalLM.from_pretrained(commands.model, use_auth_token=auth_token)
        input_names = ["input_ids"]
    elif commands.task == "text-generation" and commands.generative_model == "t5":
        model_pytorch = AutoModelForSeq2SeqLM.from_pretrained(commands.model, use_auth_token=auth_token)
        input_names = ["input_ids"]
    else:
        raise Exception(f"unknown task: {commands.task}")

    if hasattr(model_config, "type_vocab_size") and model_config.type_vocab_size == 0:
        try:
            input_names.remove("token_type_ids")
            logging.warning("Model doesn't have `token_type_ids`, removing them from `input_names`")
        except ValueError:
            pass

    logging.info(f"axis: {input_names}")

    model_pytorch.eval()
    if run_on_cuda:
        model_pytorch.cuda()

    tensor_shapes = list(zip(commands.batch_size, commands.seq_len))
    # create onnx model and compare results
    if commands.task == "text-generation" and commands.generative_model == "t5":
        pass
    else:
        onnx_model_path = os.path.join(commands.output, "model-original.onnx")
        # take optimal size
        inputs_pytorch = generate_multiple_inputs(
            batch_size=tensor_shapes[1][0],
            seq_len=tensor_shapes[1][1],
            input_names=input_names,
            device=commands.device,
            nb_inputs_to_gen=commands.warmup,
        )
        # TODO replace it by exports
        convert_to_onnx(
            model_pytorch=model_pytorch,
            output_path=onnx_model_path,
            inputs_pytorch=inputs_pytorch[0],
            quantization=commands.quantization,
            var_output_seq=commands.task in ["text-generation", "token-classification", "question-answering"],
            output_names=["output"] if commands.task != "question-answering" else ["start_logits", "end_logits"],
        )

    timings = {}

    def get_pytorch_infer(model: PreTrainedModel, cuda: bool, task: str):
        if task == "text-generation" and commands.generative_model == "t5":
            return infer_text_generation(
                model=model,
                run_on_cuda=cuda,
                min_length=commands.seq_len[0],
                max_length=commands.seq_len[0],
                num_beams=2,
            )
        if task in ["classification", "text-generation", "token-classification", "question-answering"]:
            return infer_classification_pytorch(model=model, run_on_cuda=cuda)

        raise Exception(f"unknown task: {task}")

    with torch.inference_mode():
        logging.info("running Pytorch (FP32) benchmark")
        pytorch_output, time_buffer = launch_inference(
            infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=commands.task),
            inputs=inputs_pytorch,
            nb_measures=commands.nb_measures,
        )
        timings["Pytorch (FP32)"] = time_buffer
        if run_on_cuda and not commands.fast:
            from torch.cuda.amp import autocast

            with autocast():
                engine_name = "Pytorch (FP16)"
                logging.info("running Pytorch (FP16) benchmark")
                model_pytorch_fp16 = model_pytorch.half()
                pytorch_fp16_output, time_buffer = launch_inference(
                    infer=get_pytorch_infer(model=model_pytorch_fp16, cuda=run_on_cuda, task=commands.task),
                    inputs=inputs_pytorch,
                    nb_measures=commands.nb_measures,
                )
                check_accuracy(
                    engine_name=engine_name,
                    pytorch_output=pytorch_output,
                    engine_output=pytorch_fp16_output,
                    tolerance=commands.atol,
                )
                timings[engine_name] = time_buffer
        elif commands.device == "cpu":
            logging.info("preparing Pytorch (INT-8) benchmark")
            model_pytorch = torch.quantization.quantize_dynamic(model_pytorch, {torch.nn.Linear}, dtype=torch.qint8)
            engine_name = "Pytorch (INT-8)"
            logging.info("running Pytorch (FP32) benchmark")
            pytorch_int8_output, time_buffer = launch_inference(
                infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=commands.task),
                inputs=inputs_pytorch,
                nb_measures=commands.nb_measures,
            )
            check_accuracy(
                engine_name=engine_name,
                pytorch_output=pytorch_output,
                engine_output=pytorch_int8_output,
                tolerance=commands.atol,
            )
            timings[engine_name] = time_buffer

    # create triton conf for models different from T5
    if commands.generative_model != "t5":
        if commands.task == "text-generation" and commands.generative_model == "gpt":
            conf_class: Type[Configuration] = ConfigurationDec
        else:
            pass
        
        def get_triton_output_shape(output: torch.Tensor, task: str) -> List[int]:
            triton_output_shape = list(output.shape)
            triton_output_shape[0] = -1  # dynamic batch size
            if task in ["text-generation", "token-classification", "question-answering"]:
                triton_output_shape[1] = -1  # dynamic sequence size
            return triton_output_shape

        triton_conf = conf_class(
            model_name_base=commands.name,
            dim_output=get_triton_output_shape(
                output=pytorch_output[0] if type(pytorch_output[0]) == torch.Tensor else pytorch_output[0][0],
                task=commands.task,
            ),
            nb_instance=commands.nb_instances,
            tensor_input_names=input_names,
            working_directory=commands.output,
            device=commands.device,
        )
    model_pytorch.cpu()

    logging.info("cleaning up")
    if run_on_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    # simplefy TODO 
    if "tensorrt" in commands.backend:
        logging.info("preparing TensorRT (FP16) benchmark")
        try:
            import tensorrt as trt
            from tensorrt.tensorrt import ICudaEngine, Logger, Runtime

            from landingai.backends.trt_utils import build_engine, load_engine, save_engine
        except ImportError:
            raise ImportError(
                "It seems that TensorRT is not yet installed. "
                "It is required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )
        tensorrt_path = os.path.join(commands.output, "model.plan")
        trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE if commands.verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        engine: ICudaEngine = build_engine(
            runtime=runtime,
            onnx_file_path=onnx_model_path,
            logger=trt_logger,
            min_shape=tensor_shapes[0],
            optimal_shape=tensor_shapes[1],
            max_shape=tensor_shapes[2],
            workspace_size=commands.workspace_size * 1024 * 1024,
            fp16=not commands.quantization,
            int8=commands.quantization,
        )
        save_engine(engine=engine, engine_file_path=tensorrt_path)
        # important to check the engine has been correctly serialized
        tensorrt_model: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

        if commands.task == "question-answering":
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], List[torch.Tensor]] = lambda x: list(
                tensorrt_model(x).values()
            )
        else:
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = lambda x: list(
                tensorrt_model(x).values()
            )[0]

        logging.info("running TensorRT (FP16) benchmark")
        engine_name = "TensorRT (FP16)"
        tensorrt_output, time_buffer = launch_inference(
            infer=tensorrt_inf, inputs=inputs_pytorch, nb_measures=commands.nb_measures
        )
        check_accuracy(
            engine_name=engine_name,
            pytorch_output=pytorch_output,
            engine_output=tensorrt_output,
            tolerance=commands.atol,
        )
        timings[engine_name] = time_buffer
        del engine, tensorrt_model, runtime  # delete all tensorrt objects
        gc.collect()
        triton_conf.create_configs(
            tokenizer=tokenizer, model_path=tensorrt_path, config=model_config, engine_type=EngineType.TensorRT
        )

    if "onnx" in commands.backend:
        pass
    
    if run_on_cuda:
        from torch.cuda import get_device_name

        print(f"Inference done on {get_device_name(0)}")

    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)
    print(f"Each inference engine output is within {commands.atol} tolerance compared to Pytorch output")


def entrypoint():
    args = parse_args()
    main(commands=args)


if __name__ == "__main__":
    entrypoint()
