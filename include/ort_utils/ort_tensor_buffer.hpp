#pragma once

#include <array>
#include <format>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <vector>

template <typename T, size_t Dim>
struct ort_tensor_buffer {
  ort_tensor_buffer(const Ort::MemoryInfo& mem_info,
                    const std::array<int64_t, Dim>&& tensor_shape)
      : shape{tensor_shape},
        buffer_memory(std::accumulate(tensor_shape.begin(), tensor_shape.end(),
                                      int64_t{1}, std::multiplies<int64_t>{}),
                      0.f),
        tensor{} {
    if constexpr (std::same_as<T, float>) {
      tensor = Ort::Value::CreateTensor(
          mem_info, buffer_memory.data(), buffer_memory.size() * sizeof(float),
          shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    } else if constexpr (std::same_as<T, uint16_t>) {
      tensor = Ort::Value::CreateTensor(mem_info, buffer_memory.data(),
                                        buffer_memory.size() * sizeof(uint16_t),
                                        shape.data(), shape.size(),
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    }
  }

  void copy_from(const ort_tensor_buffer& other) {
    assert(shape == other.shape);
    std::copy(other.buffer_memory.begin(), other.buffer_memory.end(),
              buffer_memory.begin());
  }

  void copy_from(std::span<const T> src) {
    if (src.size() != buffer_memory.size()) {
      throw std::invalid_argument(
          std::format("ort_tensor_buffer::{}: size mismatch ({} != {})",
                      __FUNCTION__, src.size(), buffer_memory.size()));
    }
    std::copy(src.begin(), src.end(), buffer_memory.begin());
  }
  std::array<int64_t, Dim> shape;
  std::vector<T> buffer_memory;
  Ort::Value tensor;
};