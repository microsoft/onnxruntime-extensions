# How to write custom ops

Custom Ops are based on ONNXRuntime-extensions API, especially **OrtLiteCustomOp** and **Tensor** class. C++ template metaprogramming is heavily used under the hood to provide big flexibility to the Custom Op authors on the parameter's count, type and order.

## Basic scenario

You have 2 ways to write a custom op: by writing a function, or by writing a structure.

### Custom op in the form of function

If your kernel is simple, you can use this option by just providing a function to compute the customized kernel. That function can have arbitrary number of inputs and outputs. For the inputs that are mandatory, their type would be like:

```C++
const Ort::Custom::Tensor<T>&
// or
const Ort::Custom::Tensor<T>*
```

For the inputs that are optional, their type would be like:

```C++
std::optional<const Ort::Custom::Tensor<T>*>
```

The function can also accept the pointer of **CUDAKernelContext**, where you can retrieve CUDA stream and other CUDA resources, if it requires to be run in CUDA GPU. 

The function will return the type **OrtStatusPtr**

Please refer to [negpos_def.h](https://github.com/microsoft/onnxruntime-extensions/blob/main/operators/math/cuda/negpos_def.h) as an example and [tensor_tuple.inc](https://github.com/microsoft/onnxruntime-extensions/blob/main/include/custom_op/tensor_tuple.inc) for more possible parameter types.

### Custom op in the form of structure

If the kernel is complicated and there are extra properties of the custom op, you can use this option by providing a C++ structure where you can put these properties as the structure's member variables. Besides that, you also need to provide the following member functions:

```C++
OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info)    // This function initialize the properties of the custom op

OrtStatusPtr Compute(...) const // This function computes the customized kernel.
```

The specification of the parameters of the Compute function is the same as the first way (custom op in the form of function)

## Advanced scenario

In some cases you need more control on the parameters, in this case you have to use the structure form, which you need to provide the implementations of the following member functions such as:

```C++
// By default the function will return OrtMemType::OrtMemTypeDefault for all the inputs, 
// you can provide your own implementation to specify the ith input is in CPU or GPU.
static OrtMemType GetInputMemoryType(size_t input_index) 

// You can specify input i shares the same memory with output j if possible, by allocating
// two array with same length for the pointer input_index and output_index seperately, and
// then let (*input_index)[k] = i and (*output_index)[k] = j.
// The return value is the length of the allocated array.
static size_t GetMayInplace(int** input_index, int** output_index)

// Release the allocated array from the GetMayInplace() function.
static void ReleaseMayInplace(int* input_index, int* output_index)
```