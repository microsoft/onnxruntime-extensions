# The pre/post processing code to ONNX model

Most pre and post processing of the DL models are written in Python code, when the user running the converted ONNX model with Python snippets, it would be very efficient and productive to convert these code snippets into the ONNX model, since the ONNX graph is actually a computation graph, it can represent the most programming code, theoretically.

In the onnxruntime_customops package, there is a utility to help on that. This tool is to trace the data flow in the processing code and convert all operation in the tracing logging into the ONNX graph, and merge all these graphs into in one single ONNX model. It supports the Python numeric operators and PyTorch's operation APIs (only a subset of the tensor API)

###Usage
In the onnxruntime_customops.utils, there is an API ```trace_for_onnx```, when it was fed with the input variables in Python code, it start a tracing session to log all operation starting from these variables. Also if there are some PyTorch API calls in the processing code, you need replace the import statement from ```import torch``` to ```from onnxruntime_customops import mytorch as torch```, which will enable these PyTorch API can be traced as well.

Overall, it will look like:

```python
from onnxruntime_customops.utils import trace_for_onnx
from onnxruntime_customops import mytorch as torch  # overload torch API if it is needed

# the raw input, like text, image, or ...
input_text = ...
with trace_for_onnx(input_text, names=['string_input']) as tc_sess:
    # The pre or/and post processing code starts
    ...
    ...
    ...
    output = ...
    # Save all trace objects into an ONNX model
    tc_sess.save_to_onnx('<all_in_one.onnx>', output)
```

Then the all-in-one model can be inference from the raw text directly
```python
from onnxruntime_customops.eager_op import EagerOp
# the input raw text
input_text = ...
full_model = EagerOp.from_model('<all_in_one.onnx>')
output = full_model(input_text)
print(output)
```
Or you do inference on this model with any other programming ONNXRuntime API, like C++, C#, Java and etc.
