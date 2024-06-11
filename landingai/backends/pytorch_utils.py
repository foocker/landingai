"""
Utils related to Pytorch inference.
"""
from typing import Callable, Dict,  Tuple, Union
import torch
from transformers import PreTrainedModel


def infer_classification_pytorch(
    model: PreTrainedModel, run_on_cuda: bool
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Perform Pytorch inference for classification task
    :param model: Pytorch model (transformers)
    :param run_on_cuda: True if should be ran on GPU
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        model_output = model(**inputs)  # noqa: F821
        if "logits" in model_output:
            model_output = model_output.logits.detach()
        elif "start_logits" in model_output and "end_logits" in model_output:
            start_logits = model_output.start_logits.detach()
            end_logits = model_output.end_logits.detach()
            model_output = (start_logits, end_logits)
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_output

    return infer


def infer_text_generation(
    model: PreTrainedModel, run_on_cuda: bool, min_length: int, max_length: int, num_beams: int
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Perform Pytorch inference for T5 text generation task
    :param model: Text generation model
    :param run_on_cuda: True if model should run on GPU
    :param min_length: minimum text length to be generated
    :param max_length: maximum text length to be generated
    :param num_beams: number of beams used for text generation
    :return: a function to perform inference
    """

    def infer(inputs: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        model_outputs = model.generate(
            inputs=inputs["input_ids"], min_length=min_length, max_length=max_length, num_beams=num_beams
        )
        if run_on_cuda:
            torch.cuda.synchronize()
        return model_outputs

    return infer
