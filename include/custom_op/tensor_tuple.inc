template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, const Custom::Tensor<data_type_def>*>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
  auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, const Custom::Tensor<data_type_def>&>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
  auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, std::optional<const Custom::Tensor<data_type_def>*>>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  if (ith_input < num_input) {
    tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  } else {
    std::tuple<T> current = std::tuple<T>{};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, const Custom::Span<data_type_def>*>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
  if (!reinterpret_cast<Custom::OrtTensor<data_type_def>*>(tensors.back().get())->IsCpuTensor()) {
    ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);
  }
  std::tuple<T> current = std::tuple<T>{&reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())->AsSpan()};
  auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, const Custom::Span<data_type_def>&>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
  if (!reinterpret_cast<Custom::OrtTensor<data_type_def>*>(tensors.back().get())->IsCpuTensor()) {
    ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);
  }
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())->AsSpan()};
  auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, std::optional<const Custom::Span<data_type_def>*>>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  if (ith_input < num_input) {
    tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
    if (!reinterpret_cast<Custom::OrtTensor<data_type_def>*>(tensors.back().get())->IsCpuTensor()) {
      ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);
    }
    std::tuple<T> current = std::tuple<T>{&reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())->AsSpan()};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  } else {
    std::tuple<T> current = std::tuple<T>{};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, data_type_def>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
  if (!reinterpret_cast<Custom::OrtTensor<data_type_def>*>(tensors.back().get())->IsCpuTensor()) {
    ORTX_CXX_API_THROW("scalar input could only be applied to CPU tensor", ORT_FAIL);
  }
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())->AsScalar()};
  auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, std::optional<data_type_def>>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  if (ith_input < num_input) {
    tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_input, true));
    if (!reinterpret_cast<Custom::OrtTensor<data_type_def>*>(tensors.back().get())->IsCpuTensor()) {
      ORTX_CXX_API_THROW("scalar input could only be applied to CPU tensor", ORT_FAIL);
    }
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())->AsScalar()};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  } else {
    std::tuple<T> current = std::tuple<T>{};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, Custom::Tensor<data_type_def>*>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_output, false));
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
  auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, Custom::Tensor<data_type_def>&>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_output, false));
  std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
  auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
  return std::tuple_cat(current, next);
}

template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
static typename std::enable_if<std::is_same<T, std::optional<Custom::Tensor<data_type_def>*>>::value, std::tuple<T, Ts...>>::type
CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
  if (ith_output < num_output) {
    tensors.push_back(std::make_unique<Custom::OrtTensor<data_type_def>>(*api, *context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type_def>*>(tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  } else {
    std::tuple<T> current = std::tuple<T>{};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
}
