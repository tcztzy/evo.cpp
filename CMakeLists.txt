cmake_minimum_required(VERSION 3.25)

project(
  evo2c
  VERSION 0.1.0
  DESCRIPTION "Native Evo 2 40B inference engine"
  LANGUAGES CXX)

include(CTest)
include(CheckLanguage)

option(EVO2C_CUDA "Build the CUDA backend" ON)
option(EVO2C_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(EVO2C_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(evo2c_configure_target target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  endif()

  if(EVO2C_SANITIZE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      message(FATAL_ERROR "EVO2C_SANITIZE requires a Clang or GNU compiler")
    endif()
    target_compile_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
  endif()
endfunction()

add_library(
  evo2c_core STATIC
  src/cli.cpp
  src/cpu_reference.cpp
  src/crc32.cpp
  src/mapped_file.cpp
  src/model_format.cpp
  src/sampler.cpp
  src/sequence_io.cpp
  src/status.cpp
  src/tokenizer.cpp
  src/version.cpp)
target_include_directories(
  evo2c_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
evo2c_configure_target(evo2c_core)

add_executable(evo2c src/main.cpp)
target_link_libraries(evo2c PRIVATE evo2c_core)
evo2c_configure_target(evo2c)

add_executable(evo2c-inspect src/inspect_main.cpp)
target_link_libraries(evo2c-inspect PRIVATE evo2c_core)
evo2c_configure_target(evo2c-inspect)

if(EVO2C_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "EVO2C_CUDA=ON but no CUDA compiler was found. Install CUDA 12.x or configure with -DEVO2C_CUDA=OFF for CPU-only tests.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  find_package(CUDAToolkit 12 REQUIRED)

  add_library(
    evo2c_cuda STATIC
    src/cuda/attention.cu
    src/cuda/hyena.cu
    src/cuda/model.cu
    src/cuda/ops.cu
    src/cuda/runtime.cu)
  target_include_directories(
    evo2c_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(
    evo2c_cuda
    PUBLIC evo2c_core CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cufft)
  set_target_properties(
    evo2c_cuda
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo2c_cuda
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
  if(EVO2C_WARNINGS_AS_ERRORS)
    target_compile_options(evo2c_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()

  add_executable(evo2c-cuda-smoke src/cuda/smoke.cu)
  target_link_libraries(
    evo2c-cuda-smoke
    PRIVATE evo2c_cuda)
  set_target_properties(
    evo2c-cuda-smoke
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo2c-cuda-smoke
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  add_executable(evo2c-cuda-probe tools/cuda_model_probe.cu)
  target_link_libraries(evo2c-cuda-probe PRIVATE evo2c_cuda)
  set_target_properties(
    evo2c-cuda-probe
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo2c-cuda-probe
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
  if(EVO2C_WARNINGS_AS_ERRORS)
    target_compile_options(
      evo2c-cuda-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(evo2c-cuda-probe PRIVATE stdc++fs)
  endif()

  if(BUILD_TESTING)
    add_executable(evo2c-cuda-ops-tests tests/test_cuda_ops.cu)
    target_link_libraries(evo2c-cuda-ops-tests PRIVATE evo2c_cuda)
    set_target_properties(
      evo2c-cuda-ops-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo2c-cuda-ops-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo2c-cuda-ops-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo2c-cuda-hyena-tests tests/test_cuda_hyena.cu)
    target_link_libraries(evo2c-cuda-hyena-tests PRIVATE evo2c_cuda)
    set_target_properties(
      evo2c-cuda-hyena-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo2c-cuda-hyena-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo2c-cuda-hyena-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo2c-cuda-attention-tests tests/test_cuda_attention.cu)
    target_link_libraries(evo2c-cuda-attention-tests PRIVATE evo2c_cuda)
    set_target_properties(
      evo2c-cuda-attention-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo2c-cuda-attention-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo2c-cuda-attention-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo2c-cuda-model-tests tests/test_cuda_model.cu)
    target_link_libraries(evo2c-cuda-model-tests PRIVATE evo2c_cuda)
    set_target_properties(
      evo2c-cuda-model-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo2c-cuda-model-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo2c-cuda-model-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()

    add_executable(evo2c-cuda-pipeline-tests tests/test_cuda_pipeline.cu)
    target_link_libraries(evo2c-cuda-pipeline-tests PRIVATE evo2c_cuda)
    set_target_properties(
      evo2c-cuda-pipeline-tests
      PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(
      evo2c-cuda-pipeline-tests
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
    if(EVO2C_WARNINGS_AS_ERRORS)
      target_compile_options(
        evo2c-cuda-pipeline-tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()
  endif()
endif()

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(evo2c-core-tests tests/test_core.cpp)
  target_link_libraries(evo2c-core-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-core-tests)
  add_test(NAME core COMMAND evo2c-core-tests)

  add_executable(evo2c-model-format-tests tests/test_model_format.cpp)
  target_link_libraries(evo2c-model-format-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-model-format-tests)
  add_test(NAME model_format COMMAND evo2c-model-format-tests)

  add_executable(evo2c-frontend-tests tests/test_frontend.cpp)
  target_link_libraries(evo2c-frontend-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-frontend-tests)
  add_test(NAME frontend COMMAND evo2c-frontend-tests)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(evo2c-model-format-tests PRIVATE stdc++fs)
    target_link_libraries(evo2c-frontend-tests PRIVATE stdc++fs)
  endif()

  add_executable(evo2c-cpu-reference-tests tests/test_cpu_reference.cpp)
  target_link_libraries(evo2c-cpu-reference-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-cpu-reference-tests)
  add_test(NAME cpu_reference COMMAND evo2c-cpu-reference-tests)

  add_test(
    NAME cpu_reference_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_vectors.py
      --binary $<TARGET_FILE:evo2c-cpu-reference-tests>
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
      -DEVO2C_FIXTURE_WRITER=$<TARGET_FILE:evo2c-model-format-tests>
      -DEVO2C_INSPECT_BINARY=$<TARGET_FILE:evo2c-inspect>
      -DEVO2C_FIXTURE_PATH=${CMAKE_CURRENT_BINARY_DIR}/Testing/tiny.evo2
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/model_inspect_contract.cmake)

  add_test(
    NAME cli_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO2C_BINARY=$<TARGET_FILE:evo2c>
      -DEVO2C_INSPECT_BINARY=$<TARGET_FILE:evo2c-inspect>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_contract.cmake)

  add_test(
    NAME runtime_dependencies
    COMMAND ${CMAKE_COMMAND}
      -DEVO2C_BINARY=$<TARGET_FILE:evo2c>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_dependencies.cmake)

  add_test(
    NAME converter_roundtrip
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_converter.py
      --inspector $<TARGET_FILE:evo2c-inspect>
      --config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m.yml
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/converter)
  set_tests_properties(
    converter_roundtrip
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

  if(EVO2C_CUDA)
    add_test(NAME cuda_smoke COMMAND evo2c-cuda-smoke)
    add_test(NAME cuda_ops COMMAND evo2c-cuda-ops-tests)
    add_test(NAME cuda_hyena COMMAND evo2c-cuda-hyena-tests)
    add_test(NAME cuda_attention COMMAND evo2c-cuda-attention-tests)
    add_test(
      NAME cuda_model
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo2c-cuda-model-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_model)
    add_test(
      NAME cuda_pipeline
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo2c-cuda-pipeline-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_pipeline)
    set_tests_properties(cuda_smoke PROPERTIES LABELS "gpu;cuda")
    set_tests_properties(cuda_ops PROPERTIES LABELS "gpu;cuda")
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
  endif()
endif()
