- [mmdeploy文档](https://mmdeploy.readthedocs.io/zh_CN/latest/tutorial/03_pytorch2onnx.html)基本经验，入门必看。     
- [onnxop](https://github.com/onnx/onnx/blob/main/docs/Operators.md)查找算子实现版本，和具体实现方式。  
- [torchonnxop](https://github.com/pytorch/pytorch/tree/main/torch/onnx)在onnx对应版本的symbolic_opset文件中搜索算子名称。vscode跳转到实现地方，在symbolic_fn中，通过g.op能看到算子被映射到onnx算子的情况，每个g.op是一个onnx算子。 
- [导模型]https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%AF%BC%E5%87%BA)  