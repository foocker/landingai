# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod


class Exporter(ABC):
    """Exporter base class to output model to onnx, torch_script, graphdef, etc.
    """

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def export_onnx(self, output_dir: str, opset=18, **kwargs):
        """Export the model as onnx format files.

        In some cases,  several files may be generated,
        So please return a dict which contains the generated name with the file path.

        Args:
            opset: The version of the ONNX operator set to use.
            output_dir: The output dir.
            kwargs: In this default implementation,
                kwargs will be carried to generate_dummy_inputs as extra arguments (like input shape).

        Returns:
            A dict contains the model name with the model file path.
        """
        pass
