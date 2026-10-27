# ONNX Runtime C++ Utilities

A small C++ utility library providing lightweight wrappers around [ONNX Runtime](https://onnxruntime.ai/) for session management and tensor buffer allocation.

The library consists headers only and is designed to simplify common ONNX Runtime tasks in C++ applications, particularly inference pipelines.

It also provide CMake integration for native Linux and Android build.

## Features

- **`OrtSessionHandler`** — Creates and manages an ONNX Runtime session, configures execution providers, and supports session configuration entries.
- **`ort_tensor_buffer<T, Dim>`** — Allocates and owns a tensor's backing buffer while creating an `Ort::Value` that references that memory.
- **Execution provider fallback** — Falls back to the CPU execution provider if execution provider registration throws an `Ort::Exception`.
- **Configurable sessions** — Supports execution provider options and ONNX Runtime session configuration entries.
- **FP32 and FP16 tensor storage** — Supports `float` and `uint16_t` buffers, mapped to ONNX `FLOAT` and `FLOAT16` tensors, respectively.

## Requirements

- C++20 or later
- [ONNX Runtime](https://onnxruntime.ai/) C++ API
- [magic_enum](https://github.com/Neargye/magic_enum) for execution provider naming in the session handler

The `ort_tensor_buffer` header uses `std::same_as`, which requires C++20.

## Installation

You can fetch the library with CMake's FetchContent and then links against it:

```cmake
FetchContent_Declare(
  ort_utils
  GIT_REPOSITORY https://github.com/grybouilli/ort_utils.git
  GIT_TAG main)
FetchContent_MakeAvailable(ort_utils)
# ...

target_link_libraries(my_app PRIVATE ort_utils)
```

This will link against the library as well as against the fetched ONNX Runtime library and magic_enum.

## Usage

### 1. Creating an ONNX Runtime session

`OrtSessionHandler` wraps the initialization of an ONNX Runtime session. It accepts a model filename, an execution provider name, optional execution provider options, optional session configuration entries, and an optional logging name.

```cpp
#include "OrtSessionHandler.hpp"

#include <string>
#include <unordered_map>

int main() {
    OrtSessionHandler handler(
        "model.onnx",
        "CPUExecutionProvider"
    );

    Ort::Session& session = handler.session();

    // Use session for inference.
}
```

#### Constructor parameters

| Parameter        | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `model_filename` | Path to the ONNX model file                           |
| `ep_name`        | Name of the execution provider to register            |
| `ep_options`     | Optional key-value options for the execution provider |
| `config_entries` | Optional ONNX Runtime session configuration entries   |
| `log_name`       | Name associated with the ONNX Runtime environment     |

The last three parameters are optional.

#### Using execution provider options

Execution provider options are passed as a string-to-string map:

```cpp
std::unordered_map<std::string, std::string> ep_options{
    // Add provider-specific options here.
};

OrtSessionHandler handler(
    "model.onnx",
    "CPUExecutionProvider",
    ep_options
);
```

The accepted options depend on the selected execution provider and the ONNX Runtime build.

#### Using session configuration entries

Session configuration entries can be supplied separately from execution provider options:

```cpp
std::unordered_map<std::string, std::string> config_entries{
    {"session.use_deterministic_compute", "1"}
};

OrtSessionHandler handler(
    "model.onnx",
    "CPUExecutionProvider",
    {},
    config_entries
);
```

#### Accessing the environment

The handler exposes references to both the session and its ONNX Runtime environment:

```cpp
Ort::Session& session = handler.session();
Ort::Env& env = handler.env();
```

The references remain valid while the handler and their underlying resources remain alive.

#### Execution provider fallback

If registering the requested execution provider throws an `Ort::Exception`, the handler attempts to register `CPUExecutionProvider` instead.

This fallback only covers execution provider registration. Errors encountered later, such as loading an invalid model or creating a session with incompatible provider options, can still cause construction to fail.

The fallback also depends on CPU execution provider support being available in the ONNX Runtime build.

### 2. Allocating tensor buffers

`ort_tensor_buffer<T, Dim>` allocates a zero-initialized buffer and creates an ONNX Runtime tensor referencing that buffer.

Template parameters:

| Parameter | Description                                |
| --------- | ------------------------------------------ |
| `T`       | Buffer element type: `float` or `uint16_t` |
| `Dim`     | Number of tensor dimensions                |

The constructor accepts an `Ort::MemoryInfo` and a tensor shape.

#### Example: FP32 tensor

```cpp
#include "ort_tensor_buffer.hpp"

#include <array>
#include <onnxruntime_cxx_api.h>

int main() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    ort_tensor_buffer<float, 2> input(
        memory_info,
        std::array<int64_t, 2>{2, 4}
    );

    // Tensor shape: [2, 4]
    // Buffer contains 8 zero-initialized float elements.
    Ort::Value& tensor = input.tensor;

    input.buffer_memory[0] = 1.0f;
    input.buffer_memory[1] = 2.0f;
}
```

The tensor references the buffer owned by `input`. Changes made through `buffer_memory` are visible through the ONNX tensor.

#### Example: FP16 tensor storage

```cpp
ort_tensor_buffer<uint16_t, 2> input(
    memory_info,
    std::array<int64_t, 2>{2, 4}
);
```

This creates a tensor with ONNX element type `FLOAT16` and shape `[2, 4]`.

The `uint16_t` buffer stores the raw 16-bit representations of FP16 values. It does not automatically convert ordinary floating-point values to FP16. Use an appropriate FP16 conversion routine when writing nonzero values.

#### Accessing tensor metadata

The shape is available directly from the buffer object:

```cpp
const auto& shape = input.shape;
```

The underlying ONNX tensor can also be queried:

```cpp
auto info = input.tensor.GetTensorTypeAndShapeInfo();

auto tensor_shape = info.GetShape();
auto element_type = info.GetElementType();
auto element_count = info.GetElementCount();
```

### 3. Using a tensor buffer for inference

Tensor buffers can be used directly with the ONNX Runtime C++ API:

```cpp
ort_tensor_buffer<float, 2> input_x(
    memory_info,
    std::array<int64_t, 2>{1, 4}
);

ort_tensor_buffer<float, 2> input_y(
    memory_info,
    std::array<int64_t, 2>{1, 4}
);

input_x.buffer_memory = {1.0f, 2.0f, 3.0f, 4.0f};
input_y.buffer_memory = {5.0f, 6.0f, 7.0f, 8.0f};

// Both input_x.tensor and input_y.tensor are Ort::Value objects
// ready to be passed to an ONNX Runtime inference call.
```

The model's expected input names, shapes, types, and number of inputs must match the tensors supplied to `Ort::Session::Run()`.

## Testing

The library can be tested with [Catch2](https://github.com/catchorg/Catch2).

Recommended tests include:

- Session creation with the CPU execution provider
- Inference with a small test model
- Execution provider fallback behavior
- Session configuration entry handling
- Tensor buffer allocation and zero initialization
- Tensor shape, element type, and element count
- Shared memory between the buffer and its `Ort::Value`

A minimal test model can be generated with Python and the `onnx` package. For example, an `Add` model with inputs `X` and `Y` and output `Z` is sufficient to verify session creation and inference.

## Important considerations

- **Buffer lifetime:** The `Ort::Value` created by `ort_tensor_buffer` references the vector's memory. Keep the buffer alive while using the tensor, and do not resize the vector while the tensor is in use.
- **Supported types:** The buffer implementation supports `float` and `uint16_t`. Other types are not currently supported.
- **Tensor shapes:** Supply valid dimensions and ensure the product of the dimensions fits in the supported element-count range.
- **Execution providers:** Available providers and their options depend on the ONNX Runtime build and the target platform.
- **Error handling:** Model loading and session initialization errors can propagate as ONNX Runtime exceptions.

## License

MIT License
