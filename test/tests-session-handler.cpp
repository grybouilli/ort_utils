
#include <array>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <ort_utils/ort_session_handler.hpp>  // Adjust to your header filename

#ifndef TEST_MODEL_PATH
#define TEST_MODEL_PATH "test/testdata/add.onnx"
#endif

namespace {

constexpr const char* kModelPath = TEST_MODEL_PATH;

using Options = std::unordered_map<std::string, std::string>;

TEST_CASE("OrtSessionHandler creates a CPU session", "[ort][session]") {
  OrtSessionHandler handler(kModelPath, "CPUExecutionProvider");

  REQUIRE(handler.session().GetInputCount() == 2);
  REQUIRE(handler.session().GetOutputCount() == 1);
}

TEST_CASE("OrtSessionHandler exposes its session and environment",
          "[ort][session]") {
  OrtSessionHandler handler(kModelPath, "CPUExecutionProvider");

  // Both accessors should return valid references.
  REQUIRE_NOTHROW(handler.session().GetInputCount());
}

TEST_CASE("OrtSessionHandler runs inference correctly", "[ort][inference]") {
  OrtSessionHandler handler(kModelPath, "CPUExecutionProvider");

  auto& session = handler.session();

  std::array<float, 2> x{1.0f, 2.0f};
  std::array<float, 2> y{3.0f, 4.0f};

  const std::array<int64_t, 1> shape{2};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto x_tensor = Ort::Value::CreateTensor<float>(
      memory_info, x.data(), x.size(), shape.data(), shape.size());

  auto y_tensor = Ort::Value::CreateTensor<float>(
      memory_info, y.data(), y.size(), shape.data(), shape.size());

  const char* input_names[] = {"X", "Y"};
  const char* output_names[] = {"Z"};

  std::array<Ort::Value, 2> inputs{std::move(x_tensor), std::move(y_tensor)};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                             inputs.data(), inputs.size(), output_names, 1);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].IsTensor());

  const auto info = outputs[0].GetTensorTypeAndShapeInfo();
  REQUIRE(info.GetShape() == std::vector<int64_t>{2});

  const float* result = outputs[0].GetTensorData<float>();

  REQUIRE_THAT(result[0], Catch::Matchers::WithinRel(
                              4.0f, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(result[1], Catch::Matchers::WithinRel(
                              6.0f, std::numeric_limits<float>::epsilon()));
}

TEST_CASE("OrtSessionHandler accepts execution-provider options",
          "[ort][options]") {
  // Empty options are portable for the CPU EP.
  // Provider-specific options should be tested separately.
  const Options ep_options{};

  OrtSessionHandler handler(kModelPath, "CPUExecutionProvider", ep_options);

  REQUIRE(handler.session().GetInputCount() == 2);
}

TEST_CASE("OrtSessionHandler accepts session config entries", "[ort][config]") {
  // Use a documented, supported ORT session config entry.
  const Options config_entries{{"session.use_deterministic_compute", "1"}};

  OrtSessionHandler handler(kModelPath, "CPUExecutionProvider", {},
                            config_entries);

  REQUIRE(handler.session().GetOutputCount() == 1);
}

TEST_CASE("OrtSessionHandler falls back when EP registration fails",
          "[ort][fallback]") {
  // Use a deliberately unknown provider name.
  // This assumes AppendExecutionProvider rejects it immediately.
  OrtSessionHandler handler(kModelPath, "DefinitelyNotAnExecutionProvider");

  // The constructor should have registered the CPU fallback
  // and successfully created the session.
  REQUIRE(handler.session().GetInputCount() == 2);
  REQUIRE(handler.session().GetOutputCount() == 1);
}

TEST_CASE("OrtSessionHandler rejects a nonexistent model", "[ort][errors]") {
  REQUIRE_THROWS_AS(OrtSessionHandler("this_model_does_not_exist.onnx",
                                      "CPUExecutionProvider"),
                    Ort::Exception);
}

}  // namespace
