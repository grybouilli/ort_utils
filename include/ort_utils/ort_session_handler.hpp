#pragma once

#include <filesystem>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <stdexcept>
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
    for (const auto& p : Ort::GetAvailableProviders()) {
      std::cout << "  - " << p << std::endl;
    }

    bool ep_appended = false;

#ifdef ORT_UTILS_USE_QNN_EP_PLUGIN
    // Plugin path: register the shared library, then select its devices.
    if (ep_name == magic_enum::enum_name(SupportedEPs::QNNExecutionProvider)) {
      try {
        m_env.RegisterExecutionProviderLibrary(
            kQnnRegistrationName,
            std::filesystem::path(ORT_UTILS_QNN_EP_PLUGIN_PATH).native());
        m_registered_libs.push_back(kQnnRegistrationName);

        std::vector<Ort::ConstEpDevice> selected;
        for (const auto& device : m_env.GetEpDevices()) {
          if (ep_name == device.EpName()) selected.push_back(device);
        }
        if (selected.empty()) {
          throw std::runtime_error("No devices exposed by " + ep_name +
                                   " plugin");
        }

        m_session_options.AppendExecutionProvider_V2(m_env, selected,
                                                     ep_options);
        ep_appended = true;
        std::cout << ep_name << " (plugin) registered with " << selected.size()
                  << " device(s)" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "QNN plugin EP failed: " << e.what() << std::endl;
        unregisterPlugins();
      }
    }
#endif

    // Built-in / statically linked EP path
    if (!ep_appended) {
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
    }

    m_session_options.SetIntraOpNumThreads(0);
    m_session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    m_session_options.SetLogSeverityLevel(0);
    m_session_options.SetLogId("ort_session");

    std::cout << "============= EP options =============" << std::endl;
    for (const auto& [entry, value] : ep_options) {
      std::cout << entry << ": " << value << std::endl;
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
      unregisterPlugins();
      throw;
    }
    std::cout << "Session initiated" << std::endl;
  }

  ~OrtSessionHandler() {
    m_session
        .reset();  // session must die before the plugin library is unloaded
    unregisterPlugins();
  }

  OrtSessionHandler(const OrtSessionHandler&) = delete;
  OrtSessionHandler& operator=(const OrtSessionHandler&) = delete;

  Ort::Session& session() { return m_session.value(); }
  Ort::Env& env() { return m_env; }

 private:
  static constexpr const char* kQnnRegistrationName = "QNN";

  void unregisterPlugins() {
    for (const auto& name : m_registered_libs) {
      try {
        m_env.UnregisterExecutionProviderLibrary(name.c_str());
      } catch (const Ort::Exception& e) {
        std::cerr << "Unregister " << name << " failed: " << e.what()
                  << std::endl;
      }
    }
    m_registered_libs.clear();
  }

  Ort::Env m_env;
  Ort::SessionOptions m_session_options;
  std::optional<Ort::Session> m_session;
  std::vector<std::string> m_registered_libs;
};