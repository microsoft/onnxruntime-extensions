// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <mutex>
#include <complex>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ocos_python_ARRAY_API

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <thread>
#include "string_utils.h"
#include "string_tensor.h"
#include "pykernel.h"

namespace nb = nanobind;
using namespace nb::literals;

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

struct ScopedPyBuffer {
  explicit ScopedPyBuffer(PyObject* object, int flags) {
    if (PyObject_GetBuffer(object, &view, flags) != 0) {
      throw nb::python_error();
    }
  }

  ~ScopedPyBuffer() { PyBuffer_Release(&view); }

  Py_buffer view{};
};

static nb::module_& GetNumpyModule() {
  static nb::module_ numpy = nb::module_::import_("numpy");
  return numpy;
}

static const char* GetNumpyDTypeName(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool_";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex64";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex128";
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      throw std::runtime_error(MakeString("Type float16 not supported by python customops api"));
    default:
      throw std::runtime_error(MakeString("Unsupported ONNX Tensor type: ", dtype));
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

  static nb::object BuildPyArrayFromTensor(
      const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context, const OrtValue* value,
      const shape_t& shape, ONNXTensorElementDataType dtype) {
    std::vector<std::size_t> npy_dims;
    npy_dims.reserve(shape.size());
    for (auto n : shape) {
      npy_dims.push_back(static_cast<std::size_t>(n));
    }

    auto& numpy = GetNumpyModule();
    if (dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      std::vector<std::string> src;
      GetTensorMutableDataString(api, ort, context, value, src);
      nb::list py_values;
      for (const auto& item : src) {
        py_values.append(nb::str(item.c_str()));
      }
      nb::object obj = numpy.attr("array")(py_values, "dtype"_a = numpy.attr("object_"));
      obj = obj.attr("reshape")(nb::cast(npy_dims));
      return obj;
    }

    nb::object obj = numpy.attr("empty")(nb::cast(npy_dims), "dtype"_a = GetNumpyDTypeName(dtype));
    const void* p = static_cast<const void*>(ort.GetTensorData<char>(value));
    size_t byte_size = element_size(dtype) * static_cast<size_t>(calc_size_from_shape(shape));
    ScopedPyBuffer buffer(obj.ptr(), PyBUF_WRITABLE);
    if (static_cast<size_t>(buffer.view.len) < byte_size) {
      throw std::runtime_error("Numpy buffer is smaller than tensor data.");
    }
    memcpy(buffer.view.buf, p, byte_size);
    return obj;
  }

  static nb::object InvokePyFunction(uint64_t id, const nb::object& feed, const nb::object& attrs) {
    return (*op_invoker)(id, feed, attrs);
  }

  using callback_t = std::function<nb::object(uint64_t id, const nb::object&, const nb::object&)>;
  static std::unique_ptr<callback_t> op_invoker;
};

std::unique_ptr<PyCustomOpDefImpl::callback_t> PyCustomOpDefImpl::op_invoker;
typedef struct {
  const OrtValue* input_X;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> dimensions;
} InputInformation;

PyCustomOpKernel::PyCustomOpKernel(const OrtApi& api, const OrtKernelInfo& info,
                                   uint64_t id, const std::map<std::string, int>& attrs)
    : api_(api),
      ort_(api_),
      obj_id_(id) {
  for (std::map<std::string, int>::const_iterator it = attrs.begin(); it != attrs.end(); ++it) {
    std::string attr_name = it->first;
    int attr_type = it->second;
    OrtStatus* status = nullptr;
    std::string attr_value;
    if (attr_type == PyCustomOpDef::dt_int64) {
      int64_t value = 0;
      status = api_.KernelInfoGetAttribute_int64(&info, attr_name.c_str(), &value);
      if (status == nullptr) {
        std::stringstream ss;
        ss << value;
        attr_value = ss.str();
      }
    } else if (attr_type == PyCustomOpDef::dt_float) {
      float value = 0.f;
      status = api_.KernelInfoGetAttribute_float(&info, attr_name.c_str(), &value);
      if (status == nullptr) {
        std::stringstream ss;
        ss << value;
        attr_value = ss.str();
      }
    } else if (attr_type == PyCustomOpDef::dt_string) {
      size_t size = 0;
      status = api_.KernelInfoGetAttribute_string(&info, attr_name.c_str(), nullptr, &size);
      if (status == nullptr || api_.GetErrorCode(status) == ORT_INVALID_ARGUMENT) {
        attr_value = std::string(size, ' ');
        status = api_.KernelInfoGetAttribute_string(&info, attr_name.c_str(), attr_value.data(), &size);
        if ((status != nullptr) && (api_.GetErrorCode(status) != ORT_OK)) {
          api_.ReleaseStatus(status);
          throw std::runtime_error(MakeString(
              "Unable to retrieve attribute '", attr_name, "' due to '",
              api_.GetErrorMessage(status), "'."));
        }
        if (status != nullptr) {
          api_.ReleaseStatus(status);
        }
        attr_value.resize(size - 1);
      }
    }

    if ((status != nullptr) && api_.GetErrorCode(status) != ORT_INVALID_ARGUMENT) {
      std::string error_message(api_.GetErrorMessage(status));
      api_.ReleaseStatus(status);
      throw std::runtime_error(MakeString(
          "Unable to find attribute '", attr_name, "' due to '",
          error_message, "'."));
    }
    if (status != nullptr) {
      api_.ReleaseStatus(status);
    }

    attrs_values_[attr_name] = attr_value;
  }
}

void PyCustomOpKernel::Compute(OrtKernelContext* context) {
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

  {
    /* Acquire GIL before calling Python C API, due to it was released in sess.run */
    nb::gil_scoped_acquire acquire;
    nb::list pyinputs;
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
      nb::object input0 = PyCustomOpDefImpl::BuildPyArrayFromTensor(
          api_, ort_, context, it->input_X, it->dimensions, it->dtype);
      pyinputs.append(input0);
    }

    nb::dict pyattrs;
    for (auto it = attrs_values_.begin(); it != attrs_values_.end(); ++it) {
      pyattrs[nb::cast(it->first)] = nb::cast(it->second);
    }

    // Call python function id, shape, flat coefficient.
    nb::tuple fetch = nb::cast<nb::tuple>(PyCustomOpDefImpl::InvokePyFunction(obj_id_, pyinputs, pyattrs));
    int64_t rid = nb::cast<int64_t>(fetch[0]);
    assert(rid == obj_id_);

    // Setup output.
    for (size_t no = 0; no < n_outputs; ++no) {
      auto dims = nb::cast<std::vector<int64_t>>(fetch[1 + no * 2]);
      OrtValue* output = ort_.KernelContext_GetOutput(context, no, dims.data(), dims.size());
      OrtTensorTypeAndShapeInfo* o_info = ort_.GetTensorTypeAndShape(output);
      ONNXTensorElementDataType o_dtype = ort_.GetTensorElementType(o_info);
      ort_.ReleaseTensorTypeAndShapeInfo(o_info);

      if (o_dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        std::vector<std::string> retval = nb::cast<std::vector<std::string>>(fetch[2 + no * 2]);
        FillTensorDataString(api_, ort_, context, retval, output);
      } else {
        void* out = (void*)ort_.GetTensorMutableData<float>(output);
        nb::object retval = GetNumpyModule().attr("ascontiguousarray")(fetch[2 + no * 2], "dtype"_a = GetNumpyDTypeName(o_dtype));
        ScopedPyBuffer buffer(retval.ptr(), PyBUF_CONTIG_RO);
        size_t size = element_size(o_dtype);
        size_t output_bytes = size * static_cast<size_t>(PyCustomOpDefImpl::calc_size_from_shape(dims));
        if (static_cast<size_t>(buffer.view.itemsize) != size) {
          throw std::runtime_error(MakeString(
              "Type mismatch between declared output element size (",
              size, ") and python element size (",
              buffer.view.itemsize, ")"));
        }
        if (static_cast<size_t>(buffer.view.len) != output_bytes) {
          throw std::runtime_error(MakeString(
              "Type mismatch between expected output bytes (",
              output_bytes, ") and python output bytes (",
              buffer.view.len, ")"));
        }
        memcpy(out, buffer.view.buf, output_bytes);
      }
    }
  }
}

std::map<std::string, std::vector<PyCustomOpFactory>>& PyOp_container() {
  static std::map<std::string, std::vector<PyCustomOpFactory>> map_custom_opdef;
  return map_custom_opdef;
}

void PyCustomOpDef::AddOp(const PyCustomOpDef* cod) {
  // try to fetch the domain name from op_type firstly.
  std::string op_domain = c_OpDomain;
  std::string op = cod->op_type;
  auto dm_pos = cod->op_type.find("::");
  if (std::string::npos != dm_pos) {
    op_domain = cod->op_type.substr(0, dm_pos);
    op = cod->op_type.substr(dm_pos + 2, -1);
  }

  // No need to protect against concurrent access, GIL is doing that.
  auto val = std::make_pair(op_domain, std::vector<PyCustomOpFactory>());
  auto [it_domain_op, success] = PyOp_container().insert(val);
  assert(success || !it_domain_op->second.empty());
  it_domain_op->second.emplace_back(cod, op_domain, op);
}

const PyCustomOpFactory* PyCustomOpDef_FetchPyCustomOps(size_t num) {
  if (!EnablePyCustomOps(true)) {
    EnablePyCustomOps(false);
    return nullptr;
  }

  auto it = PyOp_container().find(c_OpDomain);
  if (it != PyOp_container().end()) {
    const std::vector<PyCustomOpFactory>& ref = it->second;
    if (num < ref.size()) {
      return ref.data() + num;
    }
  }

  return nullptr;
}

const OrtCustomOp* FetchPyCustomOps(size_t& num) {
  auto ptr = PyCustomOpDef_FetchPyCustomOps(num);
  if (ptr == nullptr)  // For the breakpoint in debugging.
    return nullptr;
  return ptr;
}

bool EnablePyCustomOps(bool enabled) {
  static bool f_pyop_enabled = true;
  bool last = f_pyop_enabled;
  f_pyop_enabled = enabled;
  return last;
}

OrtStatusPtr RegisterPythonDomainAndOps(OrtSessionOptions* options, const OrtApi* ortApi) {
  OrtCustomOpDomain* domain = nullptr;
  OrtStatus* status = nullptr;

  for (auto const& val_pair : PyOp_container()) {
    if (val_pair.first == c_OpDomain) {
      continue;  // Register this domain in the second iteration.
    }

    if (status = ortApi->CreateCustomOpDomain(val_pair.first.c_str(), &domain); status) {
      return status;
    }

    for (auto const& cop : val_pair.second) {
      if (status = ortApi->CustomOpDomain_Add(domain, &cop); status) {
        return status;
      }
    }

    if (status = ortApi->AddCustomOpDomain(options, domain); status) {
      return status;
    }
  }

  return status;
}

uint64_t hash_64(const std::string& str, uint64_t num_buckets, bool fast) {
  if (fast) {
    return Hash64Fast(str.c_str(), str.size()) % static_cast<uint64_t>(num_buckets);
  }
  return Hash64(str.c_str(), str.size()) % static_cast<uint64_t>(num_buckets);
}

void AddGlobalMethods(nb::module_& m) {
  m.def("hash_64", &hash_64, "Computes a uint64 hash for a string (from tensorflow).");
  m.def("enable_py_op", &EnablePyCustomOps, "Enable or disable pyop functions.");
  m.def(
      "add_custom_op", [](const PyCustomOpDef& cod) { PyCustomOpDef::AddOp(&cod); }, "Add a PyOp Python object.");
  m.def(
      "default_opset_domain", [] { return std::string(c_OpDomain); }, "return the default opset domain name.");
}

void AddObjectMethods(nb::module_& m) {
  nb::class_<PyCustomOpDef>(m, "PyCustomOpDef")
      .def(nb::init<>())
      .def_rw("op_type", &PyCustomOpDef::op_type)
      .def_rw("obj_id", &PyCustomOpDef::obj_id)
      .def_rw("input_types", &PyCustomOpDef::input_types)
      .def_rw("output_types", &PyCustomOpDef::output_types)
      .def_rw("attrs", &PyCustomOpDef::attrs)
      .def_static("install_hooker", [](nb::object obj) {
        PyCustomOpDefImpl::op_invoker = std::make_unique<PyCustomOpDefImpl::callback_t>(
            nb::cast<PyCustomOpDefImpl::callback_t>(obj));
      })
      .def_ro_static("undefined", &PyCustomOpDef::undefined)
      .def_ro_static("dt_float", &PyCustomOpDef::dt_float)
      .def_ro_static("dt_uint8", &PyCustomOpDef::dt_uint8)
      .def_ro_static("dt_int8", &PyCustomOpDef::dt_int8)
      .def_ro_static("dt_uint16", &PyCustomOpDef::dt_uint16)
      .def_ro_static("dt_int16", &PyCustomOpDef::dt_int16)
      .def_ro_static("dt_int32", &PyCustomOpDef::dt_int32)
      .def_ro_static("dt_int64", &PyCustomOpDef::dt_int64)
      .def_ro_static("dt_string", &PyCustomOpDef::dt_string)
      .def_ro_static("dt_bool", &PyCustomOpDef::dt_bool)
      .def_ro_static("dt_float16", &PyCustomOpDef::dt_float16)
      .def_ro_static("dt_double", &PyCustomOpDef::dt_double)
      .def_ro_static("dt_uint32", &PyCustomOpDef::dt_uint32)
      .def_ro_static("dt_uint64", &PyCustomOpDef::dt_uint64)
      .def_ro_static("dt_complex64", &PyCustomOpDef::dt_complex64)
      .def_ro_static("dt_complex128", &PyCustomOpDef::dt_complex128)
      .def_ro_static("dt_bfloat16", &PyCustomOpDef::dt_bfloat16);
}

NB_MODULE(_extensions_pydll, m) {
  m.attr("__doc__") = "nanobind stateful interface to ONNXRuntime-Extensions";

  AddGlobalMethods(m);
#if defined(ENABLE_C_API)
  AddGlobalMethodsCApi(m);
#endif
  AddObjectMethods(m);
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() {
    PyCustomOpDefImpl::op_invoker.reset();
  }));
}
