import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
import tvm.contrib.graph_executor as graph_runtime
import sys
import json
from tvm.contrib.download import download_testdata
import torch
import torchvision
from transformers import DistilBertModel, DistilBertTokenizer

#####################################################
#
# This is an example of how to generate source code 
# and schedule json from tvm. 
#
#####################################################

source_file = open("mod.cu", "w")
graph_json_file = open("mod.json", "w")
host_json_file = open("host.json", "w")
param_file = open("mod.params", "w+b")

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)
model = model.eval()
class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DistilBertWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.last_hidden_state
wrapped_model = DistilBertWrapper(model)
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
print(input_ids)
scripted_model = torch.jit.trace(wrapped_model, input_ids).eval()
shape_list = [("data", input_ids.shape)]
mod, params = tvm.relay.frontend.pytorch.from_pytorch(scripted_model, shape_list)

# mod, params = relay.testing.inception_v3.get_workload(
#     batch_size=batch_size, image_shape=image_shape #(3, 299, 299)
#)
# mod, params = relay.testing.densenet.get_workload(
#     densenet_size=201, batch_size=batch_size, image_shape=image_shape #(3, 224, 224)
# )
# mod, params = relay.testing.vgg.get_workload(
#     num_layers=19, batch_size=batch_size, image_shape=image_shape #(3, 224, 224)
# )
# mod, params = relay.testing.resnet.get_workload(
#     num_layers=152, batch_size=batch_size, image_shape=image_shape #(3, 224, 224)
# )

opt_level = 3
use_cuda = tvm.runtime.enabled("cuda")
if use_cuda:
    target = tvm.target.Target("cuda -arch=sm_80")
    ctx = tvm.cuda()
else:
    target = tvm.target.Target("rocm -model=gfx908")
    ctx = tvm.rocm()

with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

source_file.write(lib.get_lib().imported_modules[0].get_source("hip"))
source_file.close()

graph_json_file.write(lib.get_graph_json())
graph_json_file.close()

module = graph_runtime.GraphModule(lib["default"](ctx))

data = np.ones(data_shape).astype("float32")
data = data * 10
module.set_input("data", input_ids)
module.run()
host_json_file.write(module.module["get_host_json"]())
host_json_file.close()

def dump_params(params, f):
    import array
    magic = bytes("TVM_MODEL_PARAMS\0", "ascii")
    f.write(magic)
    f.write(array.array('Q',[len(params)]).tobytes())
    for k in params.keys():
        param = array.array('f', params[k].asnumpy().flatten().tolist())
        f.write(bytes(k, "ascii"))
        f.write(bytes("\0", "ascii"))
        f.write(array.array('Q',[len(param)]).tobytes())
        f.write(param.tobytes())

dump_params(lib.get_params(), param_file)    
param_file.close()

batch_size = input_ids.shape[0]
sequence_length = input_ids.shape[1]
hidden_size = model.config.hidden_size
out_shape = (batch_size, sequence_length, hidden_size)
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
print(out.flatten()[0:10])