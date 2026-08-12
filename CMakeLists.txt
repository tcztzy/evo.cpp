cmake_minimum_required(VERSION 3.25)

project(
  evo_cpp
  VERSION 0.1.0
  DESCRIPTION "evo.cpp: native biological-sequence model runtime"
  LANGUAGES C CXX)

include(CTest)
include(CheckLanguage)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
find_package(Threads REQUIRED)
find_library(EVO_ZLIB_LIBRARY NAMES z zlib zlib1 libz.so.1 REQUIRED)

option(EVO_CUDA "Build the CUDA backend" ON)
option(EVO_NPY "Build the NPY output module" ${EVO_CUDA})
option(EVO_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(EVO_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(EVO_ABI_VERSION_MAJOR 1)

if(EVO_CUDA AND NOT EVO_NPY)
  message(FATAL_ERROR "EVO_CUDA requires EVO_NPY=ON")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(evo_configure_target target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()

  if(EVO_SANITIZE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      message(FATAL_ERROR "EVO_SANITIZE requires a Clang or GNU compiler")
    endif()
    target_compile_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
  endif()
endfunction()

add_library(
  evo_core STATIC
  src/cli.cpp
  src/cpu_reference.cpp
  src/cpu/esmc.cpp
  src/cpu/hyenadna.cpp
  src/cpu/model.cpp
  src/cpu/inference_cli.cpp
  src/artifact_cache.cpp
  src/backend.cpp
  src/benchmark.cpp
  src/fp8.cpp
  src/generation_io.cpp
  src/json.cpp
  src/input_stream.cpp
  src/mapped_file.cpp
  src/model_format.cpp
  src/model_registry.cpp
  src/profile.cpp
  src/sampler.cpp
  src/sequence_io.cpp
  src/status.cpp
  src/tokenizer.cpp
  src/variant.cpp
  src/variant_io.cpp
  src/version.cpp)
add_library(evo::core ALIAS evo_core)
target_include_directories(
  evo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(
  evo_core
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN ON)
evo_configure_target(evo_core)
target_link_libraries(evo_core PRIVATE "${EVO_ZLIB_LIBRARY}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(evo_core PUBLIC stdc++fs)
endif()

add_library(evo_server_core STATIC src/scheduler.cpp)
target_include_directories(
  evo_server_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(evo_server_core PRIVATE Threads::Threads)
set_target_properties(
  evo_server_core
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN ON)
evo_configure_target(evo_server_core)

add_library(evo_c SHARED src/c_api.cpp)
add_library(evo::evo ALIAS evo_c)
target_include_directories(
  evo_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(evo_c PRIVATE evo_core)
target_compile_definitions(
  evo_c
  PRIVATE EVO_BUILDING_LIBRARY=1
  PUBLIC EVO_SHARED=1)
set_target_properties(
  evo_c
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME evo
    OUTPUT_NAME evo
    VERSION ${PROJECT_VERSION}
    SOVERSION ${EVO_ABI_VERSION_MAJOR}
    VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(
    evo_c PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/evo.map")
  set_property(
    TARGET evo_c APPEND PROPERTY
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/evo.map")
endif()
evo_configure_target(evo_c)

if(EVO_NPY)
  set(EVO_LIBNPY_SOURCE_DIR "" CACHE PATH
      "Existing libnpy source tree (optional offline override)")
  if(EVO_LIBNPY_SOURCE_DIR)
    if(NOT EXISTS "${EVO_LIBNPY_SOURCE_DIR}/include/npy.hpp")
      message(FATAL_ERROR
        "EVO_LIBNPY_SOURCE_DIR does not contain include/npy.hpp")
    endif()
    add_library(evo_libnpy_headers INTERFACE)
    add_library(libnpy::libnpy ALIAS evo_libnpy_headers)
    target_include_directories(
      evo_libnpy_headers SYSTEM INTERFACE
        $<BUILD_INTERFACE:${EVO_LIBNPY_SOURCE_DIR}/include>)
  else()
    find_package(libnpy 1.0.1 CONFIG QUIET)
    if(NOT TARGET libnpy::libnpy)
      message(FATAL_ERROR
        "EVO_NPY=ON requires libnpy/1.0.1. "
        "Run 'conan install . --build=missing -s build_type=Release', then "
        "configure CMake with the generated Conan preset.")
    endif()
  endif()

  add_library(evo_npy STATIC src/npy.cpp)
  add_library(evo::npy ALIAS evo_npy)
  target_include_directories(
    evo_npy
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(
    evo_npy
    PUBLIC evo_core
    PRIVATE libnpy::libnpy)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(evo_npy PUBLIC stdc++fs)
  endif()
  set_target_properties(
    evo_npy
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      POSITION_INDEPENDENT_CODE ON
      VISIBILITY_INLINES_HIDDEN ON)
  evo_configure_target(evo_npy)
endif()

add_executable(evo src/main.cpp src/server.cpp)
target_link_libraries(
  evo PRIVATE evo_core evo_server_core evo_c Threads::Threads)
evo_configure_target(evo)

add_executable(evo-inspect src/inspect_main.cpp)
target_link_libraries(evo-inspect PRIVATE evo_core)
evo_configure_target(evo-inspect)

if(EVO_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "EVO_CUDA=ON but no CUDA compiler was found. Install CUDA 12.x or configure with -DEVO_CUDA=OFF for CPU-only tests.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  find_package(CUDAToolkit 12 REQUIRED)

  set(EVO_FLASH_ATTENTION_SOURCE_DIR "" CACHE PATH
      "Existing FlashAttention source tree (optional offline override)")
  set(EVO_CUTLASS_SOURCE_DIR "" CACHE PATH
      "Existing CUTLASS source tree (optional offline override)")
  include(FetchContent)
  if(NOT EVO_FLASH_ATTENTION_SOURCE_DIR)
    FetchContent_Declare(
      evo_flash_attention
      URL https://github.com/Dao-AILab/flash-attention/archive/628452c73a4fab560189a7caa8702642c6a38235.tar.gz
      URL_HASH SHA256=352bee729ad09f21362ed23c22dd4826deeea5acf0a00fb4a86af2863a666958
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(evo_flash_attention)
    if(NOT evo_flash_attention_POPULATED)
      FetchContent_Populate(evo_flash_attention)
    endif()
    set(EVO_FLASH_ATTENTION_SOURCE_DIR
        "${evo_flash_attention_SOURCE_DIR}")
  endif()
  if(NOT EVO_CUTLASS_SOURCE_DIR)
    FetchContent_Declare(
      evo_cutlass
      URL https://github.com/NVIDIA/cutlass/archive/7127592069c2fe01b041e174ba4345ef9b279671.tar.gz
      URL_HASH SHA256=12baa9c2bf22eaca065b5e55387c21bb4ac742992a6ecd2dd1d3b016dfd25a0a
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(evo_cutlass)
    if(NOT evo_cutlass_POPULATED)
      FetchContent_Populate(evo_cutlass)
    endif()
    set(EVO_CUTLASS_SOURCE_DIR "${evo_cutlass_SOURCE_DIR}")
  endif()
  if(NOT EXISTS
      "${EVO_FLASH_ATTENTION_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_kernel.h")
    message(FATAL_ERROR
      "EVO_FLASH_ATTENTION_SOURCE_DIR does not contain FlashAttention sources")
  endif()
  if(NOT EXISTS "${EVO_CUTLASS_SOURCE_DIR}/include/cute/tensor.hpp")
    message(FATAL_ERROR
      "EVO_CUTLASS_SOURCE_DIR does not contain CUTLASS headers")
  endif()

  add_library(
    evo_cuda STATIC
    src/cuda/attention.cu
    src/cuda/esmc.cu
    src/cuda/flash_attention.cu
    src/cuda/hyena.cu
    src/cuda/model.cu
    src/cuda/ops.cu
    src/cuda/runtime.cu)
  target_include_directories(
    evo_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  # Pinned FlashAttention/CUTLASS headers intentionally trigger host-compiler
  # warnings. Mark only those include roots as system headers, so the evo
  # wrapper itself remains under the normal -Werror policy.
  target_include_directories(
    evo_cuda SYSTEM PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash-attention-shim
      ${EVO_FLASH_ATTENTION_SOURCE_DIR}/csrc/flash_attn/src
      ${EVO_CUTLASS_SOURCE_DIR}/include)
  target_link_libraries(
    evo_cuda
    PUBLIC
      evo_core
      Threads::Threads
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
      CUDA::cufft)
  target_link_libraries(evo_cuda PRIVATE evo::npy)
  set_target_properties(
    evo_cuda
    PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CXX_VISIBILITY_PRESET hidden
      POSITION_INDEPENDENT_CODE ON
      VISIBILITY_INLINES_HIDDEN ON)
  target_compile_options(
    evo_cuda
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
  if(EVO_WARNINGS_AS_ERRORS)
    target_compile_options(evo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()

  target_sources(evo PRIVATE src/cuda/inference_cli.cpp src/cuda/hybrid_cli.cpp)
  target_link_libraries(evo PRIVATE evo_cuda evo::npy)
  target_compile_definitions(evo PRIVATE EVO_HAS_CUDA=1)
  target_link_libraries(evo_c PRIVATE evo_cuda)
  target_compile_definitions(evo_c PRIVATE EVO_HAS_CUDA=1)

  add_executable(evo-cuda-smoke src/cuda/smoke.cu)
  target_link_libraries(
    evo-cuda-smoke
    PRIVATE evo_cuda)
  set_target_properties(
    evo-cuda-smoke
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo-cuda-smoke
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_executable(evo-cuda-probe tools/cuda_model_probe.cu)
  target_link_libraries(evo-cuda-probe PRIVATE evo_cuda evo::npy)
  set_target_properties(
    evo-cuda-probe
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo-cuda-probe
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
  if(EVO_WARNINGS_AS_ERRORS)
    target_compile_options(
      evo-cuda-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(evo-cuda-probe PRIVATE stdc++fs)
  endif()

  if(BUILD_TESTING)
    add_executable(evo-cuda-ops-tests tests/test_cuda_ops.cu)
    target_link_libraries(evo-cuda-ops-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-ops-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-ops-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-ops-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo-cuda-hyena-tests tests/test_cuda_hyena.cu)
    target_link_libraries(evo-cuda-hyena-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-hyena-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-hyena-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-hyena-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo-cuda-attention-tests tests/test_cuda_attention.cu)
    target_link_libraries(evo-cuda-attention-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-attention-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-attention-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-attention-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo-cuda-model-tests tests/test_cuda_model.cu)
    target_link_libraries(evo-cuda-model-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-model-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-model-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-model-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo-cuda-pipeline-tests tests/test_cuda_pipeline.cu)
    target_link_libraries(evo-cuda-pipeline-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-pipeline-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-pipeline-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-pipeline-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo-cuda-esmc-tests tests/test_cuda_esmc.cu)
    target_link_libraries(evo-cuda-esmc-tests PRIVATE evo_cuda)
    set_target_properties(
      evo-cuda-esmc-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo-cuda-esmc-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo-cuda-esmc-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(
      evo-cuda-cli-test-bin
      src/main.cpp
      src/server.cpp
      src/cuda/inference_cli.cpp
      src/cuda/hybrid_cli.cpp)
    target_link_libraries(
      evo-cuda-cli-test-bin
      PRIVATE evo_cuda evo::npy evo_server_core evo_c Threads::Threads)
    target_compile_definitions(
      evo-cuda-cli-test-bin
      PRIVATE EVO_HAS_CUDA=1 EVO_ALLOW_TEST_FIXTURE=1)
    evo_configure_target(evo-cuda-cli-test-bin)
  endif()
endif()

set(EVO_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/evo")
configure_package_config_file(
  cmake/evoConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/evoConfig.cmake"
  INSTALL_DESTINATION "${EVO_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/evoConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(
  TARGETS evo_c
  EXPORT evoTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS evo evo-inspect RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(
  PROGRAMS tools/evo_fetch.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-fetch)
install(
  PROGRAMS tools/evo_profile_gate.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-profile-gate)
install(
  DIRECTORY include/evo
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN "json.hpp" EXCLUDE
  PATTERN "server.hpp" EXCLUDE)
install(
  FILES configs/model-registry.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo/configs")
install(
  FILES requirements-fetch.txt
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo")
install(
  EXPORT evoTargets
  FILE evoTargets.cmake
  NAMESPACE evo::
  DESTINATION "${EVO_INSTALL_CMAKEDIR}")
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/evoConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/evoConfigVersion.cmake"
  DESTINATION "${EVO_INSTALL_CMAKEDIR}")

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  if(EVO_NPY)
    add_executable(evo-npy-tests tests/test_npy.cpp)
    target_link_libraries(evo-npy-tests PRIVATE evo::npy)
    evo_configure_target(evo-npy-tests)
    add_test(NAME npy COMMAND evo-npy-tests)
  endif()

  add_executable(evo-core-tests tests/test_core.cpp)
  target_link_libraries(evo-core-tests PRIVATE evo_core)
  evo_configure_target(evo-core-tests)
  add_test(NAME core COMMAND evo-core-tests)

  add_executable(evo-c-api-tests tests/test_c_api.c)
  target_link_libraries(evo-c-api-tests PRIVATE evo::evo)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      evo-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(evo-c-api-tests PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(evo-c-api-tests PRIVATE /W4)
  endif()
  add_test(NAME c_api COMMAND evo-c-api-tests)
  add_test(
    NAME c_api_symbols
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_c_api_symbols.py
      --library $<TARGET_FILE:evo_c>)

  add_executable(evo-registry-dump tests/dump_model_registry.cpp)
  target_link_libraries(evo-registry-dump PRIVATE evo_core)
  evo_configure_target(evo-registry-dump)
  add_test(
    NAME native_registry
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native_registry.py
      --binary $<TARGET_FILE:evo-registry-dump>
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/configs/model-registry.json)

  add_executable(evo-model-format-tests tests/test_model_format.cpp)
  target_link_libraries(evo-model-format-tests PRIVATE evo_core)
  evo_configure_target(evo-model-format-tests)
  add_test(NAME model_format COMMAND evo-model-format-tests)

  add_executable(evo-frontend-tests tests/test_frontend.cpp)
  target_link_libraries(evo-frontend-tests PRIVATE evo_core)
  evo_configure_target(evo-frontend-tests)
  add_test(NAME frontend COMMAND evo-frontend-tests)

  add_executable(evo-server-core-tests tests/test_server.cpp)
  target_link_libraries(evo-server-core-tests PRIVATE evo_server_core evo_core)
  evo_configure_target(evo-server-core-tests)
  add_test(NAME server_core COMMAND evo-server-core-tests)

  add_executable(evo-cpu-reference-tests tests/test_cpu_reference.cpp)
  target_link_libraries(evo-cpu-reference-tests PRIVATE evo_core)
  evo_configure_target(evo-cpu-reference-tests)
  add_test(NAME cpu_reference COMMAND evo-cpu-reference-tests)

  add_executable(evo-cpu-model-tests tests/test_cpu_model.cpp)
  target_link_libraries(evo-cpu-model-tests PRIVATE evo_core)
  evo_configure_target(evo-cpu-model-tests)

  add_executable(evo-esmc-cpu-tests tests/test_esmc_cpu.cpp)
  target_link_libraries(evo-esmc-cpu-tests PRIVATE evo_core)
  evo_configure_target(evo-esmc-cpu-tests)
  add_test(
    NAME esmc_backend
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_backend.py
      --binary $<TARGET_FILE:evo-esmc-cpu-tests>
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_esmc.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/esmc-backend)
  set_tests_properties(
    esmc_backend
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
  add_executable(evo-cpu-cli-test-bin src/main.cpp src/server.cpp)
  target_link_libraries(
    evo-cpu-cli-test-bin
    PRIVATE evo_core evo_server_core evo_c Threads::Threads)
  target_compile_definitions(
    evo-cpu-cli-test-bin PRIVATE EVO_ALLOW_TEST_FIXTURE=1)
  evo_configure_target(evo-cpu-cli-test-bin)
  add_executable(evo-cpu-c-api-tests tests/test_cpu_c_api.c)
  target_link_libraries(evo-cpu-c-api-tests PRIVATE evo::evo)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      evo-cpu-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(evo-cpu-c-api-tests PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(evo-cpu-c-api-tests PRIVATE /W4)
  endif()
  add_executable(evo-hyenadna-c-api-tests tests/test_hyenadna_c_api.c)
  target_link_libraries(evo-hyenadna-c-api-tests PRIVATE evo::evo)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      evo-hyenadna-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(evo-hyenadna-c-api-tests PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(evo-hyenadna-c-api-tests PRIVATE /W4)
  endif()
  add_test(
    NAME cpu_backend
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_backend.py
      --binary $<TARGET_FILE:evo-cpu-model-tests>
      --c-api-binary $<TARGET_FILE:evo-cpu-c-api-tests>
      --cli-binary $<TARGET_FILE:evo-cpu-cli-test-bin>
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cpu-backend)
  set_tests_properties(
    cpu_backend PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
  add_test(
    NAME hyenadna_backend
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hyenadna_backend.py
      --binary $<TARGET_FILE:evo-cpu-cli-test-bin>
      --inspector $<TARGET_FILE:evo-inspect>
      --c-api-binary $<TARGET_FILE:evo-hyenadna-c-api-tests>
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_hyenadna.py
      --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_hyenadna_checkpoint.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/hyenadna-backend)
  set_tests_properties(
    hyenadna_backend
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")

  add_executable(evo-fp8-tests tests/test_fp8.cpp)
  target_link_libraries(evo-fp8-tests PRIVATE evo_core)
  evo_configure_target(evo-fp8-tests)
  add_test(NAME fp8 COMMAND evo-fp8-tests)

  add_test(
    NAME cpu_reference_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_vectors.py
      --binary $<TARGET_FILE:evo-cpu-reference-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/cpu_ops.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cpu_vectors.py)
  add_test(
    NAME model_vector_comparator
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_vectors.py)
  set_tests_properties(model_vector_comparator PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME model_inspect_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO_FIXTURE_WRITER=$<TARGET_FILE:evo-model-format-tests>
      -DEVO_INSPECT_BINARY=$<TARGET_FILE:evo-inspect>
      -DEVO_FIXTURE_PATH=${CMAKE_CURRENT_BINARY_DIR}/Testing/tiny.safetensors
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/model_inspect_contract.cmake)
  add_test(
    NAME c_api_model_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO_FIXTURE_WRITER=$<TARGET_FILE:evo-model-format-tests>
      -DEVO_C_API_BINARY=$<TARGET_FILE:evo-c-api-tests>
      -DEVO_FIXTURE_PATH=${CMAKE_CURRENT_BINARY_DIR}/Testing/c-api.safetensors
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c_api_model_contract.cmake)

  add_test(
    NAME install_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -DEVO_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/Testing/install-prefix
      -DEVO_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer
      -DEVO_CONSUMER_BUILD=${CMAKE_CURRENT_BINARY_DIR}/Testing/install-consumer
      -DEVO_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_contract.cmake)

  add_test(
    NAME cli_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO_BINARY=$<TARGET_FILE:evo>
      -DEVO_INSPECT_BINARY=$<TARGET_FILE:evo-inspect>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_contract.cmake)

  add_test(
    NAME runtime_dependencies
    COMMAND ${CMAKE_COMMAND}
      -DEVO_BINARY=$<TARGET_FILE:evo>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_dependencies.cmake)

  add_test(
    NAME remote_scripts_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_remote_scripts.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
    NAME quality_gate_resume
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quality_gate.py
      --quality-tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo2_quality_gate.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/quality-gate)
  add_test(
    NAME logit_comparator
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_logit_comparator.py
      --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_logits.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/logit-comparator)
  add_test(
    NAME prefill_benchmark_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prefill_benchmark_gate.py
      --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo/prefill_benchmark_gate.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/prefill-benchmark-gate)
  add_test(
    NAME profile_acceptance_gate
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_profile_gate.py
      --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo_profile_gate.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/profile-gate)
  add_test(
    NAME source_hygiene
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_source_hygiene.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
    NAME hf_fetch_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hf_fetch.py
      --tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo_fetch.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/hf-fetch)
  add_test(
    NAME release_pipeline_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_pipeline.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/release-pipeline)
  add_test(
    NAME documentation_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_documentation_contract.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME converter_roundtrip
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_converter.py
      --inspector $<TARGET_FILE:evo-inspect>
      --config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m.yml
      --config-dir ${CMAKE_CURRENT_SOURCE_DIR}/configs
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/converter)
  set_tests_properties(
    converter_roundtrip
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")

  add_test(
    NAME esmc_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_converter.py
      --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_esmc_checkpoint.py
      --inspector $<TARGET_FILE:evo-inspect>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/esmc-converter)
  set_tests_properties(
    esmc_converter_contract
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")

  add_test(
    NAME checkpoint_torch_integration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_torch_checkpoint.py)
  set_tests_properties(
    checkpoint_torch_integration
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
      SKIP_RETURN_CODE 77)

  add_test(
    NAME bionemo_mapping_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bionemo_mapping.py
      --bf16-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m-bionemo-bf16.yml
      --arc-config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m.yml)
  set_tests_properties(
    bionemo_mapping_contract
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")

  add_test(
    NAME bionemo_dcp_integration
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bionemo_dcp.py
      --config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m-bionemo-bf16.yml)
  set_tests_properties(
    bionemo_dcp_integration
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
      SKIP_RETURN_CODE 77)

  add_test(
    NAME triton_fp8_oracle
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_triton_fp8.py)
  set_tests_properties(
    triton_fp8_oracle
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
      LABELS "gpu;cuda"
      SKIP_RETURN_CODE 77)

  add_test(
    NAME vortex_reference_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vortex_reference.py)
  set_tests_properties(
    vortex_reference_contract
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
      SKIP_RETURN_CODE 77)

  if(EVO_CUDA)
    add_test(NAME cuda_smoke COMMAND evo-cuda-smoke)
    add_test(NAME cuda_ops COMMAND evo-cuda-ops-tests)
    add_test(
      NAME fp8_sass
      COMMAND ${CMAKE_COMMAND}
        -DEVO_CUDA_BINARY=$<TARGET_FILE:evo-cuda-ops-tests>
        -DEVO_CUOBJDUMP=${CUDAToolkit_BIN_DIR}/cuobjdump
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fp8_sass_contract.cmake)
    add_test(NAME cuda_hyena COMMAND evo-cuda-hyena-tests)
    add_test(NAME cuda_attention COMMAND evo-cuda-attention-tests)
    add_test(
      NAME cuda_model
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo-cuda-model-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_model
        --check-precision-metadata)
    add_test(
      NAME cuda_pipeline
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo-cuda-pipeline-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_pipeline)
    add_test(
      NAME cuda_esmc
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_backend.py
        --binary $<TARGET_FILE:evo-cuda-esmc-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_esmc.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_esmc)
    add_test(
      NAME cuda_cli
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_cli.py
        --binary $<TARGET_FILE:evo-cuda-cli-test-bin>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_cli)
    add_test(
      NAME cuda_server
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_server.py
        --binary $<TARGET_FILE:evo-cuda-cli-test-bin>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_server)
    add_test(
      NAME cuda_profile
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_profile.py
        --binary $<TARGET_FILE:evo-cuda-cli-test-bin>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --gate ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo_profile_gate.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_profile)
    add_test(
      NAME cuda_hybrid
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_hybrid.py
        --binary $<TARGET_FILE:evo-cuda-cli-test-bin>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --gate ${CMAKE_CURRENT_SOURCE_DIR}/tools/evo_profile_gate.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_hybrid)
    add_executable(evo-cuda-c-api-tests tests/test_cuda_c_api.c)
    target_link_libraries(evo-cuda-c-api-tests PRIVATE evo::evo)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(
        evo-cuda-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
      if(EVO_WARNINGS_AS_ERRORS)
        target_compile_options(evo-cuda-c-api-tests PRIVATE -Werror)
      endif()
    elseif(MSVC)
      target_compile_options(evo-cuda-c-api-tests PRIVATE /W4)
    endif()
    add_test(
      NAME cuda_c_api
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_c_api.py
        --binary $<TARGET_FILE:evo-cuda-c-api-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_c_api)
    set_tests_properties(cuda_smoke PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(cuda_ops PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(fp8_sass PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(cuda_hyena PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(cuda_attention PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(
      cuda_model
      PROPERTIES
        LABELS "gpu;cuda"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_pipeline
      PROPERTIES
        LABELS "gpu;cuda;multi_gpu"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_esmc
      PROPERTIES
        LABELS "gpu;cuda"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_cli
      PROPERTIES
        LABELS "gpu;cuda;multi_gpu"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_server
      PROPERTIES
        LABELS "gpu;cuda"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_profile
      PROPERTIES
        LABELS "gpu;cuda"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_hybrid
      PROPERTIES
        LABELS "gpu;cuda"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set_tests_properties(
      cuda_c_api
      PROPERTIES
        LABELS "gpu;cuda"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
  endif()
endif()
