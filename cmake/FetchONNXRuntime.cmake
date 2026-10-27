include(FetchContent)

# ── Configuration ──────────────────────────────────────────────────────────────
if(ONNX_USE_QNN)
    set(ONXX_RT_URL      "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android-qnn/${LIBONNXRUNTIME_VERSION}/onnxruntime-android-qnn-${LIBONNXRUNTIME_VERSION}.aar")
    set(QNN_API_URL "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip")
    set(QNN_API_DL_DIR ${CMAKE_BINARY_DIR}/import)
    set(QNN_API_ZIP_PATH ${QNN_API_DL_DIR}/qnn_api_v2.42.0.zip)
elseif(DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(ONXX_RT_URL      "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${LIBONNXRUNTIME_VERSION}/onnxruntime-android-${LIBONNXRUNTIME_VERSION}.aar")
else()
    set(ONXX_RT_URL      "https://github.com/microsoft/onnxruntime/releases/download/v${LIBONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${LIBONNXRUNTIME_VERSION}.tgz")
endif()
set(ONXX_RT_NAME     "onnxruntime")     # Name used internally by FetchContent

# ── Download & extract ─────────────────────────────────────────────────────────
FetchContent_Declare(
    ${ONXX_RT_NAME}
    URL        ${ONXX_RT_URL}
    DOWNLOAD_NAME   "onnxruntime-android.zip"
    DOWNLOAD_NO_EXTRACT FALSE            # Ensure the archive is extracted
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(${ONXX_RT_NAME})

# ── Locate the .a file inside the extracted folder ─────────────────────────────
# FetchContent extracts into: <build_dir>/_deps/mylib-src/
# Adjust the sub-path to match the actual layout of your archive.
set(ONXX_RT_SRC_DIR  "${${ONXX_RT_NAME}_SOURCE_DIR}")   # e.g. _deps/mylib-src
set(ONXX_RT_LIB_DIR  "${ONXX_RT_SRC_DIR}")   # e.g. _deps/mylib-src

# ── Create an IMPORTED target so the rest of the build uses it cleanly ─────────
if(NOT TARGET Ort::Ort)
    add_library(Ort::Ort STATIC IMPORTED GLOBAL)
    if(CMAKE_ANDROID_ARCH_ABI MATCHES "armeabi-v7a")
        set_target_properties(Ort::Ort PROPERTIES
            IMPORTED_LOCATION             "${ONXX_RT_SRC_DIR}/jni/armeabi-v7a/libonnxruntime.so"
            INTERFACE_INCLUDE_DIRECTORIES "${ONXX_RT_SRC_DIR}/headers"
        )
        set(ONXX_RT_LIB_DIR  "${ONXX_RT_SRC_DIR}/jni/armeabi-v7a")   # e.g. _deps/mylib-src
    elseif(CMAKE_ANDROID_ARCH_ABI MATCHES "arm64-v8a")
        set_target_properties(Ort::Ort PROPERTIES
            IMPORTED_LOCATION             "${ONXX_RT_SRC_DIR}/jni/arm64-v8a/libonnxruntime.so"
            INTERFACE_INCLUDE_DIRECTORIES "${ONXX_RT_SRC_DIR}/headers"
        )
        set(ONXX_RT_LIB_DIR  "${ONXX_RT_SRC_DIR}/jni/arm64-v8a")   # e.g. _deps/mylib-src
    else()
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            message(STATUS "Using host architecture (64-bit architecture)")
            set_target_properties(Ort::Ort PROPERTIES
                IMPORTED_LOCATION             "${ONXX_RT_SRC_DIR}/lib/libonnxruntime.so"
                INTERFACE_INCLUDE_DIRECTORIES "${ONXX_RT_SRC_DIR}/include"
            )
            set(ONXX_RT_LIB_DIR  "${ONXX_RT_SRC_DIR}/lib")
        else()
            message( FATAL_ERROR "Unknown Android architecture : ${CMAKE_ANDROID_ARCH_ABI} ( supported values: armeabi-v7a , arm64-v8a )")
            message( FATAL_ERROR "Unknown Host architecture : ${CMAKE_SIZEOF_VOID_P} ( supported values: 8 )")
        endif()
    endif()
endif()

message(STATUS "Ort static library : ${ONXX_RT_SRC_DIR}")
message(STATUS "Ort include dir    : ${ONXX_RT_SRC_DIR}/headers")

# Download QNN API if needed

if(QNN_API_URL AND NOT EXISTS ${QNN_API_ZIP_PATH})
    file(DOWNLOAD ${QNN_API_URL} ${QNN_API_ZIP_PATH} STATUS QNN_API_DOWNLOAD_STATUS SHOW_PROGRESS)
    execute_process(
            COMMAND unzip ${QNN_API_ZIP_PATH} 
            WORKING_DIRECTORY ${QNN_API_DL_DIR}/)
endif()