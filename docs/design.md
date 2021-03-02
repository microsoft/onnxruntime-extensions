# ONNX Custom Ops Design

[TOC]

## Guide for Contributing New Ops

### Design Principle

### How to Implement new Op

### Tips for Implementation

## Components for ONNX Custom Ops

ONNX Custom Ops wants to support pre-processing and post-processing for different areas including vision, text processing, audio. Ops in same area may share many fundamental functions and all ops depends on some basic function to exchange data with ONNXRuntime. However, there is huge difference between ops in different areas and the contributors of ops in specific area are not familiar with other areas. So we design the following components to abstract the common parts and separate the difference between ops in different areas.

```text
|-----------------------------|
|              Shared              |
|-----------------------------|
|  Text    |  Vision  |   Audio  |
|-----------------------------|
|               Core               |
|-----------------------------|
```

### Core

ONNXRuntime provides some C/C++ APIs for custom ops to exchanges data with runtime. For extension, those APIs usually is low-level and atomic. For development agility, we need an abstraction level by ourselves and `Core` is the component to abstract the low-level ONNXRuntime API. Besides ONNXRuntime APIs, `Core` are also responsible for the abstraction of system-level APIs, such as IO.

### Text/Vision/Audio

The implementation of each ops will be put into one of three components according their usage. In each component, it will category the ops into sub-component. Take `Text` as example:

```text
|-------------------------------|
| Tokenizer | ...|....| StringOp |
|-------------------------------|
|              Utils                 |
|-------------------------------|
```

`Utils` is the common fundamental function for this component, for `Text`,  it could be unicode-encoding-related function.

### Shared

Component `Shared` is responsible for registration of all ops and make our compiled shared lib could be load by ONNXRuntime. You can refer the API provided by ONNXRuntime for registration of CustomOps:

```c++
/*
    * Loads a DLL named 'library_path' and looks for this entry point:
    * OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
    * It then passes in the provided session options to this function along with the api base.
    * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
    * session options are destroyed, or if an error occurs and it is non null.
*/
ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path,
                void** library_handle);

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle) {
  API_IMPL_BEGIN

  Env::Default().LoadDynamicLibrary(library_path, library_handle);
  if (!*library_handle)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

  OrtStatus*(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

  Env::Default().GetSymbolFromLibrary(*library_handle, "RegisterCustomOps", (void**)&RegisterCustomOps);
  if (!RegisterCustomOps)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

  return RegisterCustomOps(options, OrtGetApiBase());
  API_IMPL_END
}
```

## Optimization for Binary Size

ONNX Custom Ops is not only designed for the servers but also for mobile devices. Compared with the server side, mobile side has strict requirements for resource like: CPU, Memory and Storage.  In most time, ops in ONNX Custom Ops won't take much computation and memory because most of them are only for pre-processing and post-processing. Meanwhile, for ONNX Custom Ops, cpu and memory consumption fully depends how each operator implements and we only need to do special optimization to those ops take a lot of CPU and MEM.

What we really need concern is the binary size of ONNX Custom Ops. Even each op takes a few KB and the whole shared library will be over MB. So we design the following mechanism for users to compile the ops they want into binary.

When C++ compiles the shared library, the compiler(more accurate, linker) will exclude the functions that are not referred directly or indirectly by the exported function. In ONNX Custom Ops, the main exported function is `RegisterCustomOps` and all ops need be brought into final lib should be registered here. We use a micro `#ifdef` to control the registration of the ops. If there is macro definition satisfies the judgement behind the `#ifdef`, the op will be compiled into the lib. The macro definition is controlled by CMake and user can provide the list of the ops they need. For user experience, we provide a python script for users to list all ops in an ONNX model.
