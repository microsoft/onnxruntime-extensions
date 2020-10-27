// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <fstream>
#include <mutex>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ocos_python_ARRAY_API
#include <numpy/arrayobject.h>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <thread>

#include "pykernel.h"

namespace py = pybind11;

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

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
      return sizeof(_C_float_complex);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return sizeof(_C_double_complex);
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return sizeof(std::string*);
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

  static py::object BuildPyObjFromTensor(const void* p, const shape_t& shape, ONNXTensorElementDataType dtype) {
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
      const std::string* src = (const std::string*)p;
      for (int i = 0; i < size; i++, src++) {
        outObj[i] = py::cast(*src);
      }
    } else {
      size_t size_type = element_size(dtype);
      memcpy(out_ptr, p, size_type * calc_size_from_shape(shape));
    }
    return obj;
  }

  static py::object InvokePyFunction(uint64_t id, const py::object& feed) {
    return (*op_invoker)(id, feed);
  }

  using callback_t = std::function<py::object(uint64_t id, const py::object&)>;
  static std::auto_ptr<callback_t> op_invoker;
};

std::auto_ptr<PyCustomOpDefImpl::callback_t> PyCustomOpDefImpl::op_invoker;
// static py::function g_pyfunc_caller;
// static std::mutex op_mutex;
// static std::condition_variable op_cv;
// static bool is_ready = false;

void PyCustomOpKernel::Compute(OrtKernelContext* context) {
  // std::unique_lock<std::mutex> lck(op_mutex);
  // is_ready = true;
  // op_cv.notify_all();
  //  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  if (ort_.KernelContext_GetInputCount(context) != 1)
    throw std::runtime_error("Python operator only implemented for 1 input.");
  if (ort_.KernelContext_GetOutputCount(context) != 1)
    throw std::runtime_error("Python operator only implemented for 1 output.");

  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<int64_t> i_dimensions;
  OrtTensorTypeAndShapeInfo* i_info = ort_.GetTensorTypeAndShape(input_X);
  i_dimensions = ort_.GetTensorShape(i_info);
  ONNXTensorElementDataType i_dtype = ort_.GetTensorElementType(i_info);
  const void* X = (const void*)ort_.GetTensorData<float>(input_X);
  ort_.ReleaseTensorTypeAndShapeInfo(i_info);

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
    py::object input0 = PyCustomOpDefImpl::BuildPyObjFromTensor(X, i_dimensions, i_dtype);
    auto feed = py::make_tuple(input0);
    py::tuple fetch = PyCustomOpDefImpl::InvokePyFunction(obj_id_, feed);
    int64_t rid = fetch[0].cast<int64_t>();
    assert(rid == obj_id_);
    auto dims = fetch[1].cast<std::vector<int64_t>>();

    // Setup output.
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    OrtTensorTypeAndShapeInfo* o_info = ort_.GetTensorTypeAndShape(output);
    ONNXTensorElementDataType o_dtype = ort_.GetTensorElementType(o_info);
    const void* Y = (const void*)ort_.GetTensorData<float>(output);
    ort_.ReleaseTensorTypeAndShapeInfo(o_info);
    void* out = (void*)ort_.GetTensorMutableData<float>(output);

    if (o_dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      auto retval = fetch[2].cast<std::vector<std::string>>();
      std::string* type_outPtr = (std::string*)out;
      std::string* end = type_outPtr + retval.size();
      const std::string* source = (const std::string*)retval.data();
      for (; type_outPtr != end; ++type_outPtr, ++source) {
        *type_outPtr = *source;
      }
    } else {
      py::array retval = fetch[2].cast<py::array>();
      if (element_size(o_dtype) != retval.itemsize()) {
        switch (o_dtype) {
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            retval = fetch[2].cast<py::array_t<float>>();
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

// static std::ofstream logger;
static int init_numpy() {
  import_array();
  // logger.open("./ggtest.log.txt", std::ofstream::out | std::ofstream::app);
  // logger << "first line." << std::endl;
  return 0;
}

void AddGlobalMethods(pybind11::module& m) {
  m.def("add_custom_op", [](const PyCustomOpDef& cod) { PyCustomOpDef::AddOp(&cod); });
}

void AddObjectMethods(pybind11::module& m) {
  pybind11::class_<PyCustomOpDef>(m, "PyCustomOpDef")
      .def(pybind11::init<>())
      .def_readwrite("op_type", &PyCustomOpDef::op_type)
      .def_readwrite("obj_id", &PyCustomOpDef::obj_id)
      .def_readwrite("input_types", &PyCustomOpDef::input_types)
      .def_readwrite("output_types", &PyCustomOpDef::output_types)
      .def_static("install_hooker", [](py::object obj) {
        std::auto_ptr<PyCustomOpDefImpl::callback_t> s_obj(new PyCustomOpDefImpl::callback_t(obj));
        PyCustomOpDefImpl::op_invoker = s_obj; })
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
      .def_readonly_static("dt_bfloat16 =", &PyCustomOpDef::dt_bfloat16);
}

PYBIND11_MODULE(_ortcustomops, m) {
  m.doc() = "pybind11 stateful interface to ORT Custom Ops library";
  //TODO: RegisterExceptions(m);

  init_numpy();
  AddGlobalMethods(m);
  AddObjectMethods(m);
}
