if(NOT DEFINED EVO_BUILD_DIR OR
   NOT DEFINED EVO_INSTALL_PREFIX OR
   NOT DEFINED EVO_CONSUMER_SOURCE OR
   NOT DEFINED EVO_CONSUMER_BUILD OR
   NOT DEFINED EVO_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "install contract requires build, prefix, and consumer paths")
endif()

file(REMOVE_RECURSE "${EVO_INSTALL_PREFIX}" "${EVO_CONSUMER_BUILD}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${EVO_BUILD_DIR}"
          --prefix "${EVO_INSTALL_PREFIX}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "evo install failed: ${install_error}${install_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${EVO_CONSUMER_SOURCE}"
          -B "${EVO_CONSUMER_BUILD}"
          "-DCMAKE_PREFIX_PATH=${EVO_INSTALL_PREFIX}"
          "-DCMAKE_BUILD_TYPE=${EVO_BUILD_TYPE}"
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "installed package configure failed: ${configure_error}${configure_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${EVO_CONSUMER_BUILD}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR
    "installed package build failed: ${build_error}${build_output}")
endif()

execute_process(
  COMMAND "${EVO_CONSUMER_BUILD}/evo-install-consumer"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error)
if(NOT run_result EQUAL 0)
  message(FATAL_ERROR
    "installed package runtime failed: ${run_error}${run_output}")
endif()

if(NOT EXISTS "${EVO_INSTALL_PREFIX}/include/evo/evo.h" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/bin/evo" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/bin/evo-inspect" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/bin/evo-fetch" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/python/evo/__init__.py" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/python/evo/artifact_profiles.py" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/model-registry.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-models.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-reference-patches/enformer-seq-length.patch" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/requirements-fetch.txt")
  message(FATAL_ERROR "install tree is missing public headers or CLI binaries")
endif()

execute_process(
  COMMAND "${EVO_PYTHON_EXECUTABLE}"
          "${EVO_INSTALL_PREFIX}/bin/evo-fetch" runtime --help
  RESULT_VARIABLE fetch_help_result
  OUTPUT_VARIABLE fetch_help_output
  ERROR_VARIABLE fetch_help_error)
if(NOT fetch_help_result EQUAL 0 OR
   NOT fetch_help_output MATCHES "--registry")
  message(FATAL_ERROR
    "installed evo-fetch cannot import its profile registry: "
    "${fetch_help_error}${fetch_help_output}")
endif()
