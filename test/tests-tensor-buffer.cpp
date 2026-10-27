
#include <array>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ort_utils/ort_tensor_buffer.hpp>

namespace {

Ort::MemoryInfo cpu_memory_info() {
  return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

TEST_CASE("Float tensor buffer allocates the expected number of elements",
          "[ort_tensor_buffer][float]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 2> buffer(mem_info, std::array<int64_t, 2>{3, 4});

  REQUIRE(buffer.buffer_memory.size() == 12);
  REQUIRE(buffer.shape == std::array<int64_t, 2>{3, 4});
}

TEST_CASE("Tensor buffer initializes its memory to zero",
          "[ort_tensor_buffer]") {
  auto mem_info = cpu_memory_info();

  SECTION("float") {
    ort_tensor_buffer<float, 2> buffer(mem_info, std::array<int64_t, 2>{2, 3});

    REQUIRE(buffer.buffer_memory.size() == 6);

    for (float value : buffer.buffer_memory) {
      CHECK(value == 0.0f);
    }
  }

  SECTION("uint16_t / FP16 storage") {
    ort_tensor_buffer<uint16_t, 2> buffer(mem_info,
                                          std::array<int64_t, 2>{2, 3});

    REQUIRE(buffer.buffer_memory.size() == 6);

    for (uint16_t value : buffer.buffer_memory) {
      CHECK(value == 0);
    }
  }
}

TEST_CASE("Float tensor has the expected ONNX metadata",
          "[ort_tensor_buffer][metadata]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 2> buffer(mem_info, std::array<int64_t, 2>{2, 5});

  REQUIRE(buffer.tensor.IsTensor());

  const auto info = buffer.tensor.GetTensorTypeAndShapeInfo();

  CHECK(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  CHECK(info.GetShape() == std::vector<int64_t>{2, 5});
  CHECK(info.GetElementCount() == 10);
}

TEST_CASE("FP16 tensor has the expected ONNX metadata",
          "[ort_tensor_buffer][metadata][fp16]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<uint16_t, 2> buffer(mem_info, std::array<int64_t, 2>{4, 8});

  REQUIRE(buffer.tensor.IsTensor());

  const auto info = buffer.tensor.GetTensorTypeAndShapeInfo();

  CHECK(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  CHECK(info.GetShape() == std::vector<int64_t>{4, 8});
  CHECK(info.GetElementCount() == 32);
}

TEST_CASE("Tensor buffer supports a one-dimensional shape",
          "[ort_tensor_buffer][shape]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 1> buffer(mem_info, std::array<int64_t, 1>{7});

  CHECK(buffer.buffer_memory.size() == 7);
  CHECK(buffer.tensor.GetTensorTypeAndShapeInfo().GetShape() ==
        std::vector<int64_t>{7});
}

TEST_CASE("Tensor buffer supports a three-dimensional shape",
          "[ort_tensor_buffer][shape]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 3> buffer(mem_info, std::array<int64_t, 3>{2, 3, 4});

  CHECK(buffer.buffer_memory.size() == 24);

  CHECK(buffer.tensor.GetTensorTypeAndShapeInfo().GetShape() ==
        std::vector<int64_t>{2, 3, 4});
}

TEST_CASE("Tensor writes are visible through the ONNX tensor",
          "[ort_tensor_buffer][memory]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 2> buffer(mem_info, std::array<int64_t, 2>{2, 2});

  buffer.buffer_memory = {1.0f, 2.0f, 3.0f, 4.0f};

  const float* tensor_data = buffer.tensor.GetTensorData<float>();

  REQUIRE(tensor_data != nullptr);

  CHECK(tensor_data[0] == 1.0f);
  CHECK(tensor_data[1] == 2.0f);
  CHECK(tensor_data[2] == 3.0f);
  CHECK(tensor_data[3] == 4.0f);
}

TEST_CASE("Tensor buffer supports a scalar-shaped tensor",
          "[ort_tensor_buffer][shape]") {
  auto mem_info = cpu_memory_info();

  ort_tensor_buffer<float, 0> buffer(mem_info, std::array<int64_t, 0>{});

  CHECK(buffer.buffer_memory.size() == 1);

  const auto info = buffer.tensor.GetTensorTypeAndShapeInfo();

  CHECK(info.GetShape().empty());
  CHECK(info.GetElementCount() == 1);
}

}  // namespace
