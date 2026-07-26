cmake_minimum_required(VERSION 3.25)

project(
  evo2c
  VERSION 0.1.0
  DESCRIPTION "Native Evo 2 inference engine"
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
  src/fp8.cpp
  src/mapped_file.cpp
  src/model_format.cpp
  src/model_registry.cpp
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
  find_package(Threads REQUIRED)
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
    PUBLIC
      evo2c_core
      Threads::Threads
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
      CUDA::cufft)
  set_target_properties(
    evo2c_cuda
    PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_options(
    evo2c_cuda
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra,-Wconversion,-Wshadow>)
  if(EVO2C_WARNINGS_AS_ERRORS)
    target_compile_options(evo2c_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()

  target_sources(evo2c PRIVATE src/cuda/inference_cli.cpp)
  target_link_libraries(evo2c PRIVATE evo2c_cuda)
  target_compile_definitions(evo2c PRIVATE EVO2C_HAS_CUDA=1)

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

    add_executable(
      evo2c-cuda-cli-test-bin
      src/main.cpp
      src/cuda/inference_cli.cpp)
    target_link_libraries(evo2c-cuda-cli-test-bin PRIVATE evo2c_cuda)
    target_compile_definitions(
      evo2c-cuda-cli-test-bin
      PRIVATE EVO2C_HAS_CUDA=1 EVO2C_ALLOW_TEST_FIXTURE=1)
    evo2c_configure_target(evo2c-cuda-cli-test-bin)
  endif()
endif()

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(evo2c-core-tests tests/test_core.cpp)
  target_link_libraries(evo2c-core-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-core-tests)
  add_test(NAME core COMMAND evo2c-core-tests)

  add_executable(evo2c-registry-dump tests/dump_model_registry.cpp)
  target_link_libraries(evo2c-registry-dump PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-registry-dump)
  add_test(
    NAME native_registry
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native_registry.py
      --binary $<TARGET_FILE:evo2c-registry-dump>
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/configs/model-registry.json)

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

  add_executable(evo2c-fp8-tests tests/test_fp8.cpp)
  target_link_libraries(evo2c-fp8-tests PRIVATE evo2c_core)
  evo2c_configure_target(evo2c-fp8-tests)
  add_test(NAME fp8 COMMAND evo2c-fp8-tests)

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
    NAME source_hygiene
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_source_hygiene.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME converter_roundtrip
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_converter.py
      --inspector $<TARGET_FILE:evo2c-inspect>
      --config ${CMAKE_CURRENT_SOURCE_DIR}/configs/evo2-40b-1m.yml
      --config-dir ${CMAKE_CURRENT_SOURCE_DIR}/configs
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

  if(EVO2C_CUDA)
    add_test(NAME cuda_smoke COMMAND evo2c-cuda-smoke)
    add_test(NAME cuda_ops COMMAND evo2c-cuda-ops-tests)
    add_test(
      NAME fp8_sass
      COMMAND ${CMAKE_COMMAND}
        -DEVO2C_CUDA_BINARY=$<TARGET_FILE:evo2c-cuda-ops-tests>
        -DEVO2C_CUOBJDUMP=${CUDAToolkit_BIN_DIR}/cuobjdump
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/fp8_sass_contract.cmake)
    add_test(NAME cuda_hyena COMMAND evo2c-cuda-hyena-tests)
    add_test(NAME cuda_attention COMMAND evo2c-cuda-attention-tests)
    add_test(
      NAME cuda_model
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo2c-cuda-model-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_model
        --check-precision-metadata)
    add_test(
      NAME cuda_pipeline
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_model.py
        --binary $<TARGET_FILE:evo2c-cuda-pipeline-tests>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_pipeline)
    add_test(
      NAME cuda_cli
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_cli.py
        --binary $<TARGET_FILE:evo2c-cuda-cli-test-bin>
        --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/cuda_cli)
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
      cuda_cli
      PROPERTIES
        LABELS "gpu;cuda;multi_gpu"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
  endif()
endif()
