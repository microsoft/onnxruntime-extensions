// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <fstream>
#include <mutex>
#include <complex>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ocos_python_ARRAY_API
#include <numpy/arrayobject.h>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <thread>
#include "utils/string_utils.h"
#include "pykernel.h"
#include "kernels/string_hash.hpp"
#include "kernels/string_common.h"

namespace py = pybind11;

const int PyCustomOpDef::undefined = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
const int PyCustomOpDef::dt_float = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;    // maps to c type float
const int PyCustomOpDef::dt_uint8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;    // maps to c type uint8_t
const int PyCustomOpDef::dt_int8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;      // maps to c type int8_t
const int PyCustomOpDef::dt_uint16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;  // maps to c type uint16_t
const int PyCustomOpDef::dt_int16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;    // maps to c type int16_t
const int PyCustomOpDef::dt_int32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;    // maps to c type int32_t
const int PyCustomOpDef::dt_int64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;    // maps to c type int64_t
const int PyCustomOpDef::dt_string = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;  // maps to c++ type std::string
const int PyCustomOpDef::dt_bool = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
const int PyCustomOpDef::dt_float16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
const int PyCustomOpDef::dt_double = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;  // maps to c type double
const int PyCustomOpDef::dt_uint32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;  // maps to c type uint32_t
const int PyCustomOpDef::dt_uint64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;  // maps to c type uint64_t
// complex with float32 real and imaginary components
const int PyCustomOpDef::dt_complex64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
// complex with float64 real and imaginary components
const int PyCustomOpDef::dt_complex128 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
// Non-IEEE floating-point format based on IEEE754 single-precision
const int PyCustomOpDef::dt_bfloat16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;

static int to_numpy(ONNXTensorElementDataType dt) {
  switch (dt) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return NPY_FLOAT;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return NPY_UINT8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return NPY_INT8;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return NPY_UINT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return NPY_INT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return NPY_INT32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return NPY_INT64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return NPY_BOOL;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return NPY_FLOAT16;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return NPY_DOUBLE;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return NPY_UINT32;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return NPY_UINT64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return NPY_COMPLEX64;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return NPY_COMPLEX128;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return NPY_OBJECT;
    default:
      throw std::runtime_error("No corresponding Numpy data type/Tensor data Type.");
  }
}

static size_t element_size(ONNXTensorElementDataType dt) {
  switch (dt) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return sizeof(int8_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return sizeof(int16_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return sizeof(bool);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return sizeof(double);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return sizeof(std::complex<float>);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return sizeof(std::complex<double>);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      throw std::runtime_error("OrtValue content cannot be casted into std::string*.");
    default:
      throw std::runtime_error("No corresponding Numpy data type/Tensor data Type.");
  }
}

static ONNXTensorElementDataType from_numpy(int dt) {
  switch (dt) {
    case NPY_FLOAT:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case NPY_UINT8:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case NPY_INT8:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case NPY_UINT16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case NPY_INT16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case NPY_INT32:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case NPY_INT64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case NPY_BOOL:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case NPY_FLOAT16:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case NPY_DOUBLE:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case NPY_UINT32:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case NPY_UINT64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case NPY_COMPLEX64:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    case NPY_COMPLEX128:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    case NPY_OBJECT:
    case NPY_STRING:
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    default:
      throw std::runtime_error("No corresponding ONNX data type/Tensor data Type.");
  }
}

struct PyCustomOpDefImpl : public PyCustomOpDef {
  typedef std::vector<int64_t> shape_t;
  static int64_t calc_size_from_shape(const shape_t& sp) {
    size_t c = 1;
    for (auto it = sp.begin(); it != sp.end(); ++it) {
      c *= *it;
    }
    return c;
  }

  static py::object BuildPyObjFromTensor(
      OrtApi& api, Ort::CustomOpApi& ort, OrtKernelContext* context, const OrtValue* value,
      const shape_t& shape, ONNXTensorElementDataType dtype) {
    std::vector<npy_intp> npy_dims;
    for (auto n : shape) {
      npy_dims.push_back(n);
    }
    const int numpy_type = to_numpy(dtype);
    py::object obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
        static_cast<int>(shape.size()), npy_dims.data(), numpy_type));
    void* out_ptr = static_cast<void*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

    if (dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      py::object* outObj = static_cast<py::object*>(out_ptr);
      auto size = calc_size_from_shape(shape);
      std::vector<std::string> src;
      GetTensorMutableDataString(api, ort, context, value, src);
      for (int i = 0; i < size; ++i) {
        outObj[i] = py::str(src[i]);
      }
    } else {
      const void* p = (const void*)ort.GetTensorData<char>(value);
      size_t size_type = element_size(dtype);
      memcpy(out_ptr, p, size_type * calc_size_from_shape(shape));
    }
    return obj;
  }

  static py::object InvokePyFunction(uint64_t id, const py::object& feed, const py::object& attrs) {
    return (*op_invoker)(id, feed, attrs);
  }

  using callback_t = std::function<py::object(uint64_t id, const py::object&, const py::object&)>;
  static std::unique_ptr<callback_t> op_invoker;
};

std::unique_ptr<PyCustomOpDefImpl::callback_t> PyCustomOpDefImpl::op_invoker;
// static py::function g_pyfunc_caller;
// static std::mutex op_mutex;
// static std::condition_variable op_cv;
// static bool is_ready = false;

typedef struct {
  const OrtValue* input_X;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> dimensions;
} InputInformation;

void PyCustomOpKernel::Compute(OrtKernelContext* context) {
  // std::unique_lock<std::mutex> lck(op_mutex);
  // is_ready = true;
  // op_cv.notify_all();
  // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  size_t n_inputs = ort_.KernelContext_GetInputCount(context);
  size_t n_outputs = ort_.KernelContext_GetOutputCount(context);

  // Setup inputs
  std::vector<InputInformation> inputs;
  inputs.reserve(n_inputs);
  for (size_t index = 0; index < n_inputs; ++index) {
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, index);
    std::vector<int64_t> i_dimensions;
    OrtTensorTypeAndShapeInfo* i_info = ort_.GetTensorTypeAndShape(input_X);
    i_dimensions = ort_.GetTensorShape(i_info);
    ONNXTensorElementDataType i_dtype = ort_.GetTensorElementType(i_info);
    ort_.ReleaseTensorTypeAndShapeInfo(i_info);
    inputs.push_back(InputInformation{input_X, i_dtype, i_dimensions});
  }

  /* Acquire GIL before calling Python code, due to it was released in sess.run */
  py::gil_scoped_acquire acquire;

  // TODO: Direct-Buffer-Access doesn't work for some reason.
  // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  // int64_t size = ort_.GetTensorShapeElementCount(output_info);
  // ort_.ReleaseTensorTypeAndShapeInfo(output_info);
  // py::buffer_info buf(
  //     const_cast<void *>(X),                     /* Pointer to buffer */
  //     sizeof(float),                             /* Size of one scalar */
  //     py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //     2,                                         /* Number of dimensions */
  //     {2, 3},                                    /* Buffer dimensions */
  //     {sizeof(float) * dimensions.data()[1],     /* Strides (in bytes) for each index */
  //      sizeof(float)});

  {
    py::list pyinputs;
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
      py::object input0 = PyCustomOpDefImpl::BuildPyObjFromTensor(
          api_, ort_, context, it->input_X, it->dimensions, it->dtype);
      pyinputs.append(input0);
    }

    py::dict pyattrs;
    for (auto it = attrs_values_.begin(); it != attrs_values_.end(); ++it) {
      pyattrs[py::str(it->first)] = py::str(it->second);
    }

    // Call python function id, shape, flat coefficient.
    py::tuple fetch = PyCustomOpDefImpl::InvokePyFunction(obj_id_, pyinputs, pyattrs);
    int64_t rid = fetch[0].cast<int64_t>();
    assert(rid == obj_id_);

    // Setup output.
    for (size_t no = 0; no < n_outputs; ++no) {
      auto dims = fetch[1 + no * 2].cast<std::vector<int64_t>>();
      OrtValue* output = ort_.KernelContext_GetOutput(context, no, dims.data(), dims.size());
      OrtTensorTypeAndShapeInfo* o_info = ort_.GetTensorTypeAndShape(output);
      ONNXTensorElementDataType o_dtype = ort_.GetTensorElementType(o_info);
      ort_.ReleaseTensorTypeAndShapeInfo(o_info);

      if (o_dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        std::vector<std::string> retval = fetch[2 + no * 2].cast<std::vector<std::string>>();
        FillTensorDataString(api_, ort_, context, retval, output);
      } else {
        const void* Y = (const void*)ort_.GetTensorData<float>(output);
        void* out = (void*)ort_.GetTensorMutableData<float>(output);
        py::array retval = fetch[2 + no * 2].cast<py::array>();
        if (element_size(o_dtype) != retval.itemsize()) {
          switch (o_dtype) {
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
              retval = fetch[2 + no * 2].cast<py::array_t<float>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
              retval = fetch[2 + no * 2].cast<py::array_t<uint8_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
              retval = fetch[2 + no * 2].cast<py::array_t<int8_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
              retval = fetch[2 + no * 2].cast<py::array_t<uint16_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
              retval = fetch[2 + no * 2].cast<py::array_t<int16_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
              retval = fetch[2 + no * 2].cast<py::array_t<int32_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
              retval = fetch[2 + no * 2].cast<py::array_t<int64_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
              retval = fetch[2 + no * 2].cast<py::array_t<bool>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
              throw std::runtime_error(MakeString(
                  "Type float16 not supported by python customops api"));
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
              retval = fetch[2 + no * 2].cast<py::array_t<double>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
              retval = fetch[2 + no * 2].cast<py::array_t<uint32_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
              retval = fetch[2 + no * 2].cast<py::array_t<uint64_t>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
              retval = fetch[2 + no * 2].cast<py::array_t<std::complex<float>>>();
              break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
              retval = fetch[2 + no * 2].cast<py::array_t<std::complex<double>>>();
              break;
            default:
              throw std::runtime_error(MakeString(
                  "Type mismatch between declared output element size (",
                  element_size(o_dtype), ") and python element size (",
                  retval.itemsize(), ")"));
          }
        }
        size_t size = element_size(o_dtype);
        memcpy(out, retval.data(), size * retval.size());
      }
    }

    py::gil_scoped_release release;

    // TODO: the return value from the python callback function doesn't work in pybind11&numpy.
    // py::gil_scoped_acquire acquire;
    // int64_t rid = fetch[0].cast<int64_t>();
    // assert(rid == obj_id_);
    // size_t ntp = fetch.size() - 1;
    // py::object ret_val = fetch[ntp];
    // auto p2 = PyArray_FROM_O(ret_val.ptr());
    // PyArrayObject* darray = reinterpret_cast<PyArrayObject*>(p2);
    // std::vector<int64_t> dims;
    // const int npy_type = PyArray_TYPE(darray);
    // {
    //   int ndim = PyArray_NDIM(darray);
    //   const npy_intp* npy_dims = PyArray_DIMS(darray);
    //   dims.resize(ndim);
    //   std::copy(npy_dims, npy_dims+ndim, dims.begin());
    // }

    // auto element_type = PyCustomOpDefImpl::from_numpy(npy_type);
    // OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    // float* out = ort_.GetTensorMutableData<float>(output);
    // const void* pyOut = PyData(darray, dd);

    // memcpy(out, pyOut, PyArray_NBYTES((PyArrayObject*)NULL));
  }
}

std::vector<PyCustomOpFactory>& PyCustomOpDef_python_operator_list() {
  static std::vector<PyCustomOpFactory> lst_custom_opdef;
  return lst_custom_opdef;
}

void PyCustomOpDef::AddOp(const PyCustomOpDef* cod) {
  // No need to protect against concurrent access, GIL is doing that.
  PyCustomOpDef_python_operator_list().push_back(PyCustomOpFactory(cod));
}

const PyCustomOpFactory* PyCustomOpDef_FetchPyCustomOps(size_t count) {
  if (!EnablePyCustomOps(true)) {
    EnablePyCustomOps(false);
    return nullptr;
  }

  // The result must stay alive
  std::vector<PyCustomOpFactory>& copy = PyCustomOpDef_python_operator_list();
  if (count < copy.size())
    return &(copy[count]);
  return nullptr;
}

const OrtCustomOp* FetchPyCustomOps(size_t& count) {
  auto ptr = PyCustomOpDef_FetchPyCustomOps(count);
  if (ptr == nullptr)
    return nullptr;
  return ptr;
}

bool EnablePyCustomOps(bool enabled) {
  static bool f_pyop_enabled = true;
  bool last = f_pyop_enabled;
  f_pyop_enabled = enabled;
  return last;
}

// static std::ofstream logger;
static int init_numpy() {
  import_array();
  // logger.open("./ggtest.log.txt", std::ofstream::out | std::ofstream::app);
  // logger << "first line." << std::endl;
  return 0;
}

uint64_t hash_64(const std::string& str, uint64_t num_buckets, bool fast) {
  if (fast) {
    return Hash64Fast(str.c_str(), str.size()) % static_cast<uint64_t>(num_buckets);
  }
  return Hash64(str.c_str(), str.size()) % static_cast<uint64_t>(num_buckets);
}

void AddGlobalMethods(pybind11::module& m) {
  m.def("enable_custom_op", &EnablePyCustomOps, "Enable or disable pyop functions.");
  m.def("add_custom_op", [](const PyCustomOpDef& cod) { PyCustomOpDef::AddOp(&cod); });
  m.def("hash_64", &hash_64, "Computes a uint64 hash for a string (from tensorflow).");
  m.def("default_opset_domain", []{return std::string(c_OpDomain);}, "return the default opset domain name");
}

void AddObjectMethods(pybind11::module& m) {
  pybind11::class_<PyCustomOpDef>(m, "PyCustomOpDef")
      .def(pybind11::init<>())
      .def_readwrite("op_type", &PyCustomOpDef::op_type)
      .def_readwrite("obj_id", &PyCustomOpDef::obj_id)
      .def_readwrite("input_types", &PyCustomOpDef::input_types)
      .def_readwrite("output_types", &PyCustomOpDef::output_types)
      .def_readwrite("attrs", &PyCustomOpDef::attrs)
      .def_static("install_hooker", [](py::object obj) { PyCustomOpDefImpl::op_invoker = std::make_unique<PyCustomOpDefImpl::callback_t>(obj); })
      .def_readonly_static("undefined", &PyCustomOpDef::undefined)
      .def_readonly_static("dt_float", &PyCustomOpDef::dt_float)
      .def_readonly_static("dt_uint8", &PyCustomOpDef::dt_uint8)
      .def_readonly_static("dt_int8", &PyCustomOpDef::dt_int8)
      .def_readonly_static("dt_uint16", &PyCustomOpDef::dt_uint16)
      .def_readonly_static("dt_int16", &PyCustomOpDef::dt_int16)
      .def_readonly_static("dt_int32", &PyCustomOpDef::dt_int32)
      .def_readonly_static("dt_int64", &PyCustomOpDef::dt_int64)
      .def_readonly_static("dt_string", &PyCustomOpDef::dt_string)
      .def_readonly_static("dt_bool", &PyCustomOpDef::dt_bool)
      .def_readonly_static("dt_float16", &PyCustomOpDef::dt_float16)
      .def_readonly_static("dt_double", &PyCustomOpDef::dt_double)
      .def_readonly_static("dt_uint32", &PyCustomOpDef::dt_uint32)
      .def_readonly_static("dt_uint64", &PyCustomOpDef::dt_uint64)
      .def_readonly_static("dt_complex64", &PyCustomOpDef::dt_complex64)
      .def_readonly_static("dt_complex128", &PyCustomOpDef::dt_complex128)
      .def_readonly_static("dt_bfloat16", &PyCustomOpDef::dt_bfloat16);
}

PYBIND11_MODULE(_ortcustomops, m) {
  m.doc() = "pybind11 stateful interface to ORT Custom Ops library";
  //TODO: RegisterExceptions(m);

  init_numpy();
  AddGlobalMethods(m);
  AddObjectMethods(m);
}
