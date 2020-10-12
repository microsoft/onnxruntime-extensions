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

const std::map<int, int>& PyCustomOpDef::get_numpy_type_map(bool from_or_to) {
  static std::map<int, int> to_type_map{
      {dt_bool, NPY_BOOL},
      {dt_float, NPY_FLOAT},
      {dt_float16, NPY_FLOAT16},
      {dt_double, NPY_DOUBLE},
      {dt_int8, NPY_INT8},
      {dt_uint8, NPY_UINT8},
      {dt_int16, NPY_INT16},
      {dt_uint16, NPY_UINT16},
      {dt_int32, NPY_INT},
      {dt_uint32, NPY_UINT},
      {dt_int64, NPY_LONGLONG},
      {dt_uint64, NPY_ULONGLONG},
  };

  static auto from_type_map = [] {std::map<int, int> reversed;
                          for(auto it:to_type_map) reversed[it.second] = it.first; return reversed; }();

  return from_or_to ? from_type_map : to_type_map;
}

struct PyCustomOpDefImpl : public PyCustomOpDef {
  static int to_numpy(int dt, bool from_or_to = false) {
    auto type_map = get_numpy_type_map(from_or_to);
    const auto it = type_map.find(dt);
    if (it == type_map.end()) {
      throw std::runtime_error("No corresponding Numpy data type/Tensor data Type.");
    } else {
      return it->second;
    }
  }

  typedef std::vector<int64_t> shape_t;
  static int64_t calc_size_from_shape(const shape_t& sp) {
    size_t c = 1;
    for (auto it = sp.begin(); it != sp.end(); ++it) {
      c *= *it;
    }
    return c;
  }

  static int from_numpy(int dt) {
    return to_numpy(dt, true);
  }

  template <typename _DT>
  static py::object BuildPyObjFromTensor(const _DT* p, const shape_t& shape) {
    std::vector<npy_intp> npy_dims;
    for (auto n : shape) {
      npy_dims.push_back(n);
    }

    const int numpy_type = to_numpy(dt_float);
    auto obj = py::reinterpret_borrow<py::object>(PyArray_SimpleNew(
        static_cast<int>(shape.size()), npy_dims.data(), numpy_type));

    void* outPtr = static_cast<void*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

    memcpy(outPtr, p, sizeof(_DT) * calc_size_from_shape(shape));
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

  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  std::vector<int64_t> dimensions;
  OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(input_X);
  dimensions = (ort_.GetTensorShape(info));
  ort_.ReleaseTensorTypeAndShapeInfo(info);

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
    py::object input0 = PyCustomOpDefImpl::BuildPyObjFromTensor(X, dimensions);
    auto feed = py::make_tuple(input0);
    py::tuple fetch = PyCustomOpDefImpl::InvokePyFunction(obj_id_, feed);
    int64_t rid = fetch[0].cast<int64_t>();
    assert(rid == obj_id_);
    auto dims = fetch[1].cast<std::vector<int64_t>>();
    auto retval = fetch[2].cast<std::vector<float>>();

    py::gil_scoped_release release;
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    float* out = ort_.GetTensorMutableData<float>(output);
    std::copy(retval.data(), retval.data()+retval.size(), out);

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

const OrtCustomOp* FetchPyCustomOps(size_t& count) {
  static std::vector<PyCustomOpFactory> c_pycustomops;
  c_pycustomops.clear();

  for (auto od_ptr : PyCustomOpDef::FullList()) {
    c_pycustomops.emplace_back(PyCustomOpFactory(od_ptr));
  }

  count = c_pycustomops.size();
  return c_pycustomops.data();
}

// static std::ofstream logger;
static int init_numpy() {
  import_array();
  // logger.open("./ggtest.log.txt", std::ofstream::out | std::ofstream::app);
  // logger << "first line." << std::endl;
  return 0;
}

void AddGlobalMethods(pybind11::module& m) {
  m.def("add_custom_op", [](const PyCustomOpDef& cod) { PyCustomOpDef::FullList().push_back(&cod); });
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
