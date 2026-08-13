if(NOT DEFINED EVO_FIXTURE_WRITER OR
   NOT DEFINED EVO_INSPECT_BINARY OR
   NOT DEFINED EVO_FIXTURE_PATH)
  message(FATAL_ERROR "model inspect contract test requires writer, inspector, and fixture paths")
endif()

get_filename_component(fixture_directory "${EVO_FIXTURE_PATH}" DIRECTORY)
file(MAKE_DIRECTORY "${fixture_directory}")
set(esmc_fixture "${EVO_FIXTURE_PATH}.esmc")

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}" --help
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error)
if(NOT help_result EQUAL 0 OR
   NOT help_output MATCHES "biological-model runtime Safetensors")
  message(FATAL_ERROR "evo-inspect help has stale project scope: ${help_error}${help_output}")
endif()

execute_process(
  COMMAND "${EVO_FIXTURE_WRITER}" --write-fixture "${EVO_FIXTURE_PATH}"
  RESULT_VARIABLE write_result
  OUTPUT_VARIABLE write_output
  ERROR_VARIABLE write_error)
if(NOT write_result EQUAL 0)
  message(FATAL_ERROR "fixture writer failed: ${write_error}${write_output}")
endif()

execute_process(
  COMMAND "${EVO_FIXTURE_WRITER}" --write-esmc-fixture "${esmc_fixture}"
  RESULT_VARIABLE esmc_write_result
  OUTPUT_VARIABLE esmc_write_output
  ERROR_VARIABLE esmc_write_error)
if(NOT esmc_write_result EQUAL 0)
  message(FATAL_ERROR "ESMC fixture writer failed: ${esmc_write_error}${esmc_write_output}")
endif()

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}" "${EVO_FIXTURE_PATH}"
  RESULT_VARIABLE inspect_result
  OUTPUT_VARIABLE inspect_output
  ERROR_VARIABLE inspect_error)
if(NOT inspect_result EQUAL 0)
  message(FATAL_ERROR "evo-inspect failed: ${inspect_error}${inspect_output}")
endif()
if(NOT inspect_output MATCHES "format=SAFETENSORS profile=evo2-runtime-v1.*validation=ok" OR
   NOT inspect_output MATCHES "metadata model.name type=string value=tiny-evo2" OR
   NOT inspect_output MATCHES "exact_support=unknown evidence=none" OR
   NOT inspect_output MATCHES "tensor_count=2" OR
   NOT inspect_output MATCHES "tensor embed.weight dtype=BF16 shape=\\[2,2\\]")
  message(FATAL_ERROR "evo-inspect output contract failed: ${inspect_output}")
endif()

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}" "${esmc_fixture}"
  RESULT_VARIABLE esmc_inspect_result
  OUTPUT_VARIABLE esmc_inspect_output
  ERROR_VARIABLE esmc_inspect_error)
if(NOT esmc_inspect_result EQUAL 0 OR
   NOT esmc_inspect_output MATCHES "profile=esmc-runtime-v1.*validation=ok" OR
   NOT esmc_inspect_output MATCHES "exact_support=validated evidence=esmc-official-oracle/2026-08-12/esmc_300m")
  message(FATAL_ERROR "evo-inspect omitted ESMC support: ${esmc_inspect_error}${esmc_inspect_output}")
endif()

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}" "${EVO_FIXTURE_PATH}" --tensor blocks.0.scale
  RESULT_VARIABLE tensor_result
  OUTPUT_VARIABLE tensor_output
  ERROR_VARIABLE tensor_error)
if(NOT tensor_result EQUAL 0 OR
   NOT tensor_output MATCHES "tensor blocks.0.scale dtype=F32 shape=\\[2\\]")
  message(FATAL_ERROR "tensor selection contract failed: ${tensor_error}${tensor_output}")
endif()

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}" "${EVO_FIXTURE_PATH}" --tensor missing.weight
  RESULT_VARIABLE missing_result
  OUTPUT_VARIABLE missing_output
  ERROR_VARIABLE missing_error)
if(missing_result EQUAL 0 OR NOT missing_error MATCHES "invalid_argument: tensor 'missing.weight' not found")
  message(FATAL_ERROR "missing tensor error contract failed: ${missing_error}${missing_output}")
endif()

file(REMOVE "${EVO_FIXTURE_PATH}")
file(REMOVE "${esmc_fixture}")
