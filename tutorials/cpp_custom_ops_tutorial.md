# Making Custom Ops with C++

This is just an outline. Full instructions will be added later.

1. Clone and build the repo by following the Getting Stated instructions [here](https://github.com/microsoft/ort-customops#getting-started).
2. Create a kernel for your custom op in the /ocos/kernels directory. Create a .cc and .hpp file.
3. Write tests in the /test directory.

To use the op in a tensorflow model, follow the [TF2ONNX Custom Ops Tutorial](https://github.com/microsoft/ort-customops/blob/main/tutorials/tf2onnx_custom_ops_tutorial.ipynb)