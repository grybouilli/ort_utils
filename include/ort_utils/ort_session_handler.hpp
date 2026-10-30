#pragma once

// #include <cpu_provider_factory.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_session_options_config_keys.h"
#include <magic_enum/magic_enum.hpp>

enum class SupportedEPs {
  CPUExecutionProvider,
  QNNExecutionProvider,
  XnnpackExecutionProvider,
  WebGpuExecutionProvider,
  NnapiExecutionProvider
};

class OrtSessionHandler {
 public:
  OrtSessionHandler(
      const std::string model_filename, const std::string ep_name,
      const std::unordered_map<std::string, std::string>& ep_options = {},
      const std::unordered_map<std::string, std::string>& config_entries = {},
      const std::string log_name = "my_ort_session\n")
      : m_env(ORT_LOGGING_LEVEL_FATAL, log_name.c_str()) {
    // Lists all providers compiled into your ORT build
    std::vector<std::string> providers = Ort::GetAvailableProviders();

    for (const auto& p : providers) {
      std::cout << "  - " << p << std::endl;
    }

    try {
      m_session_options.AppendExecutionProvider(ep_name, ep_options);
      std::cout << ep_name << " registered" << std::endl;
    } catch (const Ort::Exception& e) {
      std::cout << "Exception caught : " << e.what() << std::endl;
      m_session_options.AppendExecutionProvider("CPUExecutionProvider");
      std::cout << "Falling back to "
                << magic_enum::enum_name(SupportedEPs::CPUExecutionProvider)
                << std::endl;
    }

    m_session_options.SetIntraOpNumThreads(0);
    m_session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    m_session_options.SetLogSeverityLevel(0);
    m_session_options.SetLogId("ort_session");

    std::cout << "============= EP options =============" << std::endl;

    for (const auto& [entry, value] : ep_options) {
      std::cout << "" << entry << ": " << value << std::endl;
    }
    std::cout << "======================================" << std::endl;

    for (const auto& [entry, value] : config_entries) {
      std::cout << "Adding " << entry << ": " << value << std::endl;
      m_session_options.AddConfigEntry(entry.c_str(), value.c_str());
    }
    try {
      m_session =
          Ort::Session(m_env, model_filename.c_str(), m_session_options);
    } catch (const Ort::Exception& e) {
      std::cerr << e.what() << std::endl;
      throw e;
    }
    std::cout << "Session initiated" << std::endl;
  }

  Ort::Session& session() { return m_session.value(); }
  Ort::Env& env() { return m_env; }

 private:
  Ort::Env m_env;
  Ort::SessionOptions m_session_options;
  std::optional<Ort::Session> m_session;
};