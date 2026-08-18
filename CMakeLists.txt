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
set(EVO_MPS_PLATFORM_SUPPORTED OFF)
if(APPLE)
  if(CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set(EVO_MPS_PLATFORM_SUPPORTED ON)
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(EVO_MPS_PLATFORM_SUPPORTED ON)
  endif()
endif()
set(EVO_MPS_DEFAULT OFF)
if(EVO_MPS_PLATFORM_SUPPORTED)
  set(EVO_MPS_DEFAULT ON)
endif()
option(EVO_MPS "Build the macOS arm64 MPS backend" ${EVO_MPS_DEFAULT})
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
  src/cpu/geneb_bert.cpp
  src/cpu/geneb_custom_encoder.cpp
  src/cpu/geneb_decoder.cpp
  src/cpu/geneb_decoder_omnina_apple.cpp
  src/cpu/geneb_esm.cpp
  src/cpu/fft_convolution.cpp
  src/cpu/geneb_evo1.cpp
  src/cpu/geneb_gpt.cpp
  src/cpu/geneb_hyenadna.cpp
  src/cpu/geneb_janusdna.cpp
  src/cpu/geneb_mamba.cpp
  src/cpu/geneb_olmo.cpp
  src/cpu/geneb_roformer.cpp
  src/cpu/geneb_sequence_cnn.cpp
  src/cpu/hyenadna.cpp
  src/cpu/mamba_primitives.cpp
  src/cpu/model.cpp
  src/cpu/inference_cli.cpp
  src/artifact_cache.cpp
  src/backend.cpp
  src/benchmark.cpp
  src/fp8.cpp
  src/geneb_input_transform.cpp
  src/geneb_embedding.cpp
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
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(
    src/cpu/geneb_decoder_omnina_apple.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(evo_core PRIVATE "${EVO_ZLIB_LIBRARY}")
if(APPLE)
  find_library(EVO_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(evo_core PRIVATE "${EVO_ACCELERATE_FRAMEWORK}")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(evo_core PUBLIC stdc++fs)
endif()

if(EVO_MPS)
  if(NOT EVO_MPS_PLATFORM_SUPPORTED)
    message(FATAL_ERROR "EVO_MPS requires macOS on arm64")
  endif()
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS OFF)
  find_library(EVO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(EVO_METAL_FRAMEWORK Metal REQUIRED)
  find_library(EVO_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(EVO_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

  add_library(evo_mps STATIC src/mps/model.mm src/mps/inference_cli.cpp)
  add_library(evo::mps ALIAS evo_mps)
  target_include_directories(
    evo_mps
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(
    evo_mps
    PUBLIC evo_core
    PRIVATE
      "${EVO_FOUNDATION_FRAMEWORK}"
      "${EVO_METAL_FRAMEWORK}"
      "${EVO_MPS_FRAMEWORK}"
      "${EVO_COREGRAPHICS_FRAMEWORK}")
  target_compile_options(
    evo_mps
    PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-fvisibility=hidden>)
  set_target_properties(
    evo_mps
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      POSITION_INDEPENDENT_CODE ON
      VISIBILITY_INLINES_HIDDEN ON)
  evo_configure_target(evo_mps)
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

if(EVO_MPS)
  target_link_libraries(evo PRIVATE evo::mps)
  target_compile_definitions(evo PRIVATE EVO_HAS_MPS=1)
  target_link_libraries(evo_c PRIVATE evo::mps)
  target_compile_definitions(evo_c PRIVATE EVO_HAS_MPS=1)
endif()

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
  PROGRAMS tools/convert_tokenizer_asset.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-tokenizer)
install(
  PROGRAMS tools/convert_geneb_decoder_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-decoder)
install(
  PROGRAMS tools/convert_geneb_bert_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-bert)
install(
  PROGRAMS tools/convert_geneb_custom_encoder_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-custom-encoder)
install(
  PROGRAMS tools/convert_geneb_esm_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-esm)
install(
  PROGRAMS tools/convert_geneb_olmo_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-olmo)
install(
  PROGRAMS tools/convert_geneb_mamba_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-mamba)
install(
  PROGRAMS tools/convert_geneb_janusdna_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-janusdna)
install(
  PROGRAMS tools/convert_geneb_sequence_cnn_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-sequence-cnn)
install(
  PROGRAMS tools/convert_geneb_roformer_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-roformer)
install(
  PROGRAMS tools/convert_geneb_deepgene_tokenizer.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-deepgene-tokenizer)
install(
  PROGRAMS tools/convert_geneb_gpt2_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-gpt2)
install(
  PROGRAMS tools/convert_geneb_dna_gpt_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-dna-gpt)
install(
  PROGRAMS tools/convert_geneb_hyenadna_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-hyenadna)
install(
  PROGRAMS tools/convert_geneb_evo1_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-convert-geneb-evo1)
install(
  PROGRAMS tools/generate_dna_gpt_tokenizer_source.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-generate-dna-gpt-tokenizer)
install(
  PROGRAMS tools/run_geneb.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-run-geneb)
install(
  PROGRAMS tools/run_geneb_checkpoint_evidence.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME evo-geneb-checkpoint-evidence)
install(
  FILES
    tools/evo/__init__.py
    tools/evo/artifact_profiles.py
    tools/evo/geneb_artifact.py
    tools/evo/geneb_gpt_artifact.py
    tools/evo/geneb_long_hyena_artifact.py
    tools/evo/geneb_t36_artifact.py
    tools/evo/hf_checkpoint.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo/python/evo")
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
  FILES
    configs/model-registry.json
    configs/geneb-models.json
    configs/geneb-benchmark-spec.json
    configs/geneb-probe-lock.json
    configs/geneb-bert-models.json
    configs/geneb-custom-encoder-models.json
    configs/geneb-decoder-models.json
    configs/geneb-esm-models.json
    configs/geneb-gpt2-models.json
    configs/geneb-dna-gpt-models.json
    configs/geneb-olmo-models.json
    configs/geneb-mamba-models.json
    configs/geneb-janusdna-models.json
    configs/geneb-hyenadna-models.json
    configs/geneb-evo1-models.json
    configs/geneb-sequence-cnn-models.json
    configs/geneb-roformer-models.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo/configs")
install(
  DIRECTORY configs/tokenizers/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo/configs/tokenizers")
install(
  DIRECTORY configs/geneb-reference-patches/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/evo/configs/geneb-reference-patches")
install(
  FILES requirements-fetch.txt requirements-convert.txt requirements-geneb.txt
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

  add_executable(evo-geneb-input-transform-tests
                 tests/test_geneb_input_transform.cpp)
  target_link_libraries(evo-geneb-input-transform-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-input-transform-tests)
  add_test(
    NAME geneb_input_transform
    COMMAND evo-geneb-input-transform-tests
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json)

  add_executable(evo-geneb-decoder-tests tests/test_geneb_decoder.cpp)
  target_link_libraries(evo-geneb-decoder-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-decoder-tests)
  add_test(NAME geneb_decoder COMMAND evo-geneb-decoder-tests)
  add_executable(evo-geneb-omnina-operator-audit
                 tests/geneb_omnina_operator_audit.cpp)
  target_link_libraries(evo-geneb-omnina-operator-audit PRIVATE evo_core)
  evo_configure_target(evo-geneb-omnina-operator-audit)
  add_test(
    NAME geneb_decoder_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_decoder_vectors.py
      --binary $<TARGET_FILE:evo-geneb-decoder-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_decoder_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_decoder_vectors.py)
  set_tests_properties(geneb_decoder_vectors PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-bert-tests tests/test_geneb_bert.cpp)
  target_link_libraries(evo-geneb-bert-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-bert-tests)
  add_test(NAME geneb_bert COMMAND evo-geneb-bert-tests)
  add_test(
    NAME geneb_bert_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_bert_vectors.py
      --binary $<TARGET_FILE:evo-geneb-bert-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_bert_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_bert_vectors.py)
  set_tests_properties(geneb_bert_vectors PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-custom-encoder-tests
                 tests/test_geneb_custom_encoder.cpp)
  target_link_libraries(evo-geneb-custom-encoder-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-custom-encoder-tests)
  add_test(NAME geneb_custom_encoder COMMAND evo-geneb-custom-encoder-tests)
  add_test(
    NAME geneb_custom_encoder_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_custom_encoder_vectors.py
      --binary $<TARGET_FILE:evo-geneb-custom-encoder-tests>
      --fixture
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_custom_encoder_oracle.json
      --generator
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_custom_encoder_vectors.py)
  set_tests_properties(geneb_custom_encoder_vectors PROPERTIES
                       SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-esm-tests tests/test_geneb_esm.cpp)
  target_link_libraries(evo-geneb-esm-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-esm-tests)
  add_test(NAME geneb_esm COMMAND evo-geneb-esm-tests)
  add_test(
    NAME geneb_esm_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_esm_vectors.py
      --binary $<TARGET_FILE:evo-geneb-esm-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_esm_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_esm_vectors.py)
  set_tests_properties(geneb_esm_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_esm_torch_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_esm_vectors.py
      --binary $<TARGET_FILE:evo-geneb-esm-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_esm_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_esm_vectors.py
      --require-torch)
  set_tests_properties(geneb_esm_torch_vectors PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-olmo-tests tests/test_geneb_olmo.cpp)
  target_link_libraries(evo-geneb-olmo-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-olmo-tests)
  add_test(NAME geneb_olmo COMMAND evo-geneb-olmo-tests)
  add_test(
    NAME geneb_olmo_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_olmo_vectors.py
      --binary $<TARGET_FILE:evo-geneb-olmo-tests>)
  set_tests_properties(geneb_olmo_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_olmo_torch_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_olmo_vectors.py
      --binary $<TARGET_FILE:evo-geneb-olmo-tests> --require-torch)
  set_tests_properties(geneb_olmo_torch_vectors PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-long-hyena-tests
                 tests/test_geneb_long_hyena.cpp)
  target_link_libraries(evo-geneb-long-hyena-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-long-hyena-tests)
  add_test(NAME geneb_long_hyena COMMAND evo-geneb-long-hyena-tests)
  add_test(
    NAME geneb_long_hyena_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_long_hyena_vectors.py
      --binary $<TARGET_FILE:evo-geneb-long-hyena-tests>
      --fixture
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_long_hyena_oracle.json
      --generator
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_long_hyena_vectors.py)
  set_tests_properties(geneb_long_hyena_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME geneb_long_hyena_complexity
           COMMAND evo-geneb-long-hyena-tests --complexity)

  add_executable(evo-geneb-mamba-tests tests/test_geneb_mamba.cpp)
  target_link_libraries(evo-geneb-mamba-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-mamba-tests)
  add_test(NAME geneb_mamba COMMAND evo-geneb-mamba-tests)
  add_test(
    NAME geneb_mamba_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_mamba_vectors.py
      --binary $<TARGET_FILE:evo-geneb-mamba-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_mamba_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_mamba_vectors.py)
  set_tests_properties(geneb_mamba_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_mamba_torch_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_mamba_vectors.py
      --torch-check)
  set_tests_properties(geneb_mamba_torch_vectors PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(evo-geneb-janusdna-tests tests/test_geneb_janusdna.cpp)
  target_link_libraries(evo-geneb-janusdna-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-janusdna-tests)
  add_test(NAME geneb_janusdna COMMAND evo-geneb-janusdna-tests)
  add_test(
    NAME geneb_janusdna_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_janusdna_vectors.py
      --binary $<TARGET_FILE:evo-geneb-janusdna-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_janusdna_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_janusdna_vectors.py)
  set_tests_properties(geneb_janusdna_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_janusdna_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_janusdna_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_janusdna_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-janusdna-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-janusdna-tests>)

  add_executable(evo-geneb-sequence-cnn-tests
                 tests/test_geneb_sequence_cnn.cpp)
  target_link_libraries(evo-geneb-sequence-cnn-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-sequence-cnn-tests)
  add_test(NAME geneb_sequence_cnn COMMAND evo-geneb-sequence-cnn-tests)
  add_test(NAME geneb_sequence_cnn_complexity
           COMMAND evo-geneb-sequence-cnn-tests --complexity)
  add_test(
    NAME geneb_sequence_cnn_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_sequence_cnn_vectors.py
      --binary $<TARGET_FILE:evo-geneb-sequence-cnn-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_sequence_cnn_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_sequence_cnn_vectors.py)
  set_tests_properties(geneb_sequence_cnn_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_sequence_cnn_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_sequence_cnn_converter.py
      --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_sequence_cnn_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-sequence-cnn-models.json)

  add_executable(evo-geneb-roformer-tests tests/test_geneb_roformer.cpp)
  target_link_libraries(evo-geneb-roformer-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-roformer-tests)
  add_test(NAME geneb_roformer COMMAND evo-geneb-roformer-tests)
  add_test(NAME geneb_roformer_complexity
           COMMAND evo-geneb-roformer-tests --complexity)
  add_test(
    NAME geneb_roformer_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_roformer_vectors.py
      --binary $<TARGET_FILE:evo-geneb-roformer-tests>
      --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_roformer_oracle.json
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_roformer_vectors.py)
  set_tests_properties(geneb_roformer_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_roformer_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_roformer_converter.py
      --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_roformer_checkpoint.py
      --tokenizer-converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_deepgene_tokenizer.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-roformer-models.json)

  add_executable(evo-geneb-gpt-tests tests/test_geneb_gpt.cpp)
  target_link_libraries(evo-geneb-gpt-tests PRIVATE evo_core)
  evo_configure_target(evo-geneb-gpt-tests)
  add_test(NAME geneb_gpt COMMAND evo-geneb-gpt-tests)
  add_test(
    NAME geneb_gpt_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_gpt_vectors.py
      --binary $<TARGET_FILE:evo-geneb-gpt-tests>)
  set_tests_properties(geneb_gpt_vectors PROPERTIES SKIP_RETURN_CODE 77)
  add_test(
    NAME geneb_gpt_torch_vectors
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_gpt_vectors.py
      --binary $<TARGET_FILE:evo-geneb-gpt-tests> --require-torch)
  set_tests_properties(geneb_gpt_torch_vectors PROPERTIES SKIP_RETURN_CODE 77)

  if(EVO_MPS)
    add_executable(evo-mps-tests tests/test_mps.cpp)
    target_include_directories(evo-mps-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(evo-mps-tests PRIVATE evo::mps Threads::Threads)
    evo_configure_target(evo-mps-tests)
    add_test(NAME v42_mps_linear COMMAND evo-mps-tests)
    set_tests_properties(v42_mps_linear PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(evo-mps-c-api-tests tests/test_mps_c_api.c)
    target_link_libraries(evo-mps-c-api-tests PRIVATE evo::evo)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(
        evo-mps-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
      if(EVO_WARNINGS_AS_ERRORS)
        target_compile_options(evo-mps-c-api-tests PRIVATE -Werror)
      endif()
    endif()
  endif()

  add_executable(evo-c-api-tests tests/test_c_api.c)
  target_link_libraries(evo-c-api-tests PRIVATE evo::evo)
  if(EVO_MPS)
    target_compile_definitions(evo-c-api-tests PRIVATE EVO_TEST_HAS_MPS=1)
  endif()
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
    NAME v41_c_api_symbols
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
  add_test(
    NAME geneb_surface_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_surface_contract.py
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/configs/model-registry.json
      --c-api $<TARGET_FILE:evo-c-api-tests>
      --cli $<TARGET_FILE:evo>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-surface)
  set_tests_properties(
    geneb_surface_contract
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")

  add_executable(evo-model-format-tests tests/test_model_format.cpp)
  target_link_libraries(evo-model-format-tests PRIVATE evo_core)
  evo_configure_target(evo-model-format-tests)
  add_test(NAME model_format COMMAND evo-model-format-tests)

  add_executable(evo-tokenizer-asset-tests tests/test_tokenizer_assets.cpp)
  target_link_libraries(evo-tokenizer-asset-tests PRIVATE evo_core)
  evo_configure_target(evo-tokenizer-asset-tests)
  add_test(NAME tokenizer_assets COMMAND evo-tokenizer-asset-tests)

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
  add_executable(evo-esmc-c-api-tests tests/test_esmc_c_api.c)
  target_link_libraries(evo-esmc-c-api-tests PRIVATE evo::evo)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      evo-esmc-c-api-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(EVO_WARNINGS_AS_ERRORS)
      target_compile_options(evo-esmc-c-api-tests PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(evo-esmc-c-api-tests PRIVATE /W4)
  endif()
  add_test(
    NAME esmc_backend
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_backend.py
      --binary $<TARGET_FILE:evo-esmc-cpu-tests>
      --c-api-binary $<TARGET_FILE:evo-esmc-c-api-tests>
      --cli-binary $<TARGET_FILE:evo-cpu-cli-test-bin>
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
  if(EVO_MPS)
    target_link_libraries(evo-cpu-cli-test-bin PRIVATE evo::mps)
    target_compile_definitions(evo-cpu-cli-test-bin PRIVATE EVO_HAS_MPS=1)
  endif()
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
    NAME tokenizer_cpu_binding
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizer_cpu_binding.py
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
      --binary $<TARGET_FILE:evo-cpu-model-tests>
      --c-api-binary $<TARGET_FILE:evo-c-api-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/tokenizer-cpu-binding)
  set_tests_properties(
    tokenizer_cpu_binding
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools")
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

  if(EVO_MPS)
    add_test(
      NAME v42_mps_backend
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mps_backend.py
        --cli-binary $<TARGET_FILE:evo-cpu-cli-test-bin>
        --c-api-binary $<TARGET_FILE:evo-mps-c-api-tests>
        --tiny-generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_model.py
        --hyenadna-generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_hyenadna.py
        --esmc-generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tiny_esmc.py
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/mps-backend)
    set_tests_properties(
      v42_mps_backend
      PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/tools"
        SKIP_RETURN_CODE 77)
  endif()

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
      -DEVO_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_contract.cmake)

  add_test(
    NAME cli_contract
    COMMAND ${CMAKE_COMMAND}
      -DEVO_BINARY=$<TARGET_FILE:evo>
      -DEVO_INSPECT_BINARY=$<TARGET_FILE:evo-inspect>
      -DEVO_GENEB_CATALOG=${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
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
    NAME esmc_official_oracle_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_official_oracle.py
      --comparator ${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_esmc_oracle.py
      --generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_esmc_official_oracle.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/esmc-official-oracle)
  add_test(
    NAME esmc_benchmark_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_esmc_benchmark.py
      --official-tool ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_esmc_official.py
      --summarizer ${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize_esmc_benchmark.py
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/esmc-benchmark)
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
      --registry ${CMAKE_CURRENT_SOURCE_DIR}/configs/model-registry.json
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/hf-fetch)
  add_test(
    NAME geneb_catalog_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_catalog.py
      --validator ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_geneb_models.py
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-catalog)
  add_test(
    NAME geneb_runner_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_runner.py
      --runner ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_geneb.py
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --benchmark-spec
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-benchmark-spec.json
      --lock ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-probe-lock.json
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-runner)
  add_test(
    NAME geneb_checkpoint_evidence_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_checkpoint_evidence.py
      --tool
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_geneb_checkpoint_evidence.py
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-checkpoint-evidence)
  add_test(
    NAME tokenizer_asset_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizer_asset_converter.py
      --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_tokenizer_asset.py
      --runtime $<TARGET_FILE:evo-tokenizer-asset-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/tokenizer-asset-converter)
  add_test(
    NAME geneb_decoder_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_decoder_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_decoder_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-decoder-models.json
      --native-loader $<TARGET_FILE:evo-geneb-decoder-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-decoder-converter)
  add_test(
    NAME geneb_bert_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_bert_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_bert_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-bert-models.json
      --native-loader $<TARGET_FILE:evo-geneb-bert-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-bert-converter)
  add_test(
    NAME geneb_custom_encoder_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_custom_encoder_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_custom_encoder_checkpoint.py
      --profiles
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-custom-encoder-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-custom-encoder-tests>
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-custom-encoder-converter)
  add_test(
    NAME geneb_esm_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_esm_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_esm_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-esm-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-esm-tests>
      --c-api $<TARGET_FILE:evo-c-api-tests>
      --cli $<TARGET_FILE:evo-cpu-cli-test-bin>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-esm-converter)
  add_test(
    NAME geneb_olmo_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_olmo_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_olmo_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-olmo-models.json
      --native-loader $<TARGET_FILE:evo-geneb-olmo-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-olmo-converter)
  add_test(
    NAME geneb_mamba_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_mamba_converter.py
      --converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_mamba_checkpoint.py
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-mamba-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-mamba-tests>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-mamba-converter)
  add_test(
    NAME geneb_long_hyena_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_long_hyena_converter.py
      --hyena-converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_hyenadna_checkpoint.py
      --evo-converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_evo1_checkpoint.py
      --hyena-profiles
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-hyenadna-models.json
      --evo-profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-evo1-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-long-hyena-tests>
      --tokenizer-compiler
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_tokenizer_asset.py
      --tokenizer-runtime $<TARGET_FILE:evo-tokenizer-asset-tests>
      --oracle
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/geneb_long_hyena_oracle.json
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-long-hyena-converter)
  add_test(
    NAME geneb_gpt_converter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_gpt_converter.py
      --gpt2-converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_gpt2_checkpoint.py
      --dna-converter
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_geneb_dna_gpt_checkpoint.py
      --gpt2-profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-gpt2-models.json
      --dna-profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-dna-gpt-models.json
      --catalog ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-models.json
      --native-loader $<TARGET_FILE:evo-geneb-gpt-tests>
      --cli $<TARGET_FILE:evo>
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-gpt-converter)
  add_test(
    NAME dna_gpt_tokenizer_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dna_gpt_tokenizer.py
      --python ${Python3_EXECUTABLE}
      --generator
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_dna_gpt_tokenizer_source.py
      --compiler ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_tokenizer_asset.py
      --runtime $<TARGET_FILE:evo-tokenizer-asset-tests>
      --profiles ${CMAKE_CURRENT_SOURCE_DIR}/configs/geneb-dna-gpt-models.json
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Testing/dna-gpt-tokenizer)
  add_test(
    NAME geneb_generator_tokenizer_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geneb_generator_tokenizer.py
      --generator
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_generator_tokenizer_source.py
      --oracle-3b
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_geneb_generator_3b_upstream_oracle.py
      --manifest-1-2b
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/tokenizers/geneb-generator-eukaryote-1-2b-kmer-v1.json
      --manifest-3b
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/tokenizers/geneb-generator-eukaryote-3b-kmer-v1.json
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/Testing/geneb-generator-tokenizer)
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
        --cuda-c-api-binary $<TARGET_FILE:evo-esmc-c-api-tests>
        --cuda-cli-binary $<TARGET_FILE:evo-cuda-cli-test-bin>
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
