# from collections import OrderedDict
# from export import TorchModelExporter

# model = None
# dummy_inputs = None
# dynamic_axis = {0: 'batch', 1: 'sequence'}
# inputs = OrderedDict([
#     ('input_ids', dynamic_axis),
#     ('attention_mask', dynamic_axis),
#     ('token_type_ids', dynamic_axis),
# ])
# outputs = OrderedDict({'logits': {0: 'batch'}})
# output_files = TorchModelExporter().export_onnx(model=model, dummy_inputs=dummy_inputs, inputs=inputs, 
#                                                 outputs=outputs, output_dir='./tmp')
# print(output_files)


# from landingai.infer.trtm.trt_infer import infer_v3

# model_plan_fp16 = '/data/sadtalker_simplify/model_generator_original.plan'
# model_plan_int8 = '/data/sadtalker_simplify/model_generator_original_int8.plan'

# infer_v3(model_plan_fp16, mode='fp16')
# infer_v3(model_plan_int8, mode='int8')

# from landingai.apis import a

# print(a.ac)
