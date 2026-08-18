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
   NOT EXISTS "${EVO_INSTALL_PREFIX}/bin/evo-run-geneb" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/bin/evo-geneb-checkpoint-evidence" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/python/evo/__init__.py" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/python/evo/artifact_profiles.py" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/model-registry.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-models.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-benchmark-spec.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-probe-lock.json" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/configs/geneb-reference-patches/enformer-seq-length.patch" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/requirements-fetch.txt" OR
   NOT EXISTS "${EVO_INSTALL_PREFIX}/share/evo/requirements-geneb.txt")
  message(FATAL_ERROR "install tree is missing public headers or CLI binaries")
endif()

execute_process(
  COMMAND "${EVO_PYTHON_EXECUTABLE}"
          "${EVO_INSTALL_PREFIX}/bin/evo-geneb-checkpoint-evidence" --help
  RESULT_VARIABLE evidence_help_result
  OUTPUT_VARIABLE evidence_help_output
  ERROR_VARIABLE evidence_help_error)
if(NOT evidence_help_result EQUAL 0 OR
   NOT evidence_help_output MATCHES "canonical-checkpoint evidence")
  message(FATAL_ERROR
    "installed GENEB checkpoint evidence runner is unusable: "
    "${evidence_help_error}${evidence_help_output}")
endif()

execute_process(
  COMMAND "${EVO_PYTHON_EXECUTABLE}"
          "${EVO_INSTALL_PREFIX}/bin/evo-run-geneb" --help
  RESULT_VARIABLE geneb_help_result
  OUTPUT_VARIABLE geneb_help_output
  ERROR_VARIABLE geneb_help_error)
if(NOT geneb_help_result EQUAL 0 OR
   NOT geneb_help_output MATCHES "100-task GENEB v4")
  message(FATAL_ERROR
    "installed evo-run-geneb cannot locate its command contract: "
    "${geneb_help_error}${geneb_help_output}")
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
