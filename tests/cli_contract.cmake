if(NOT DEFINED EVO2C_BINARY OR NOT DEFINED EVO2C_INSPECT_BINARY)
  message(FATAL_ERROR "CLI contract test requires both executable paths")
endif()

execute_process(
  COMMAND "${EVO2C_BINARY}" --version
  RESULT_VARIABLE version_result
  OUTPUT_VARIABLE version_output
  ERROR_VARIABLE version_error)
if(NOT version_result EQUAL 0 OR NOT version_output MATCHES "^evo2c 0\\.1\\.0")
  message(FATAL_ERROR "evo2c --version contract failed: ${version_error}${version_output}")
endif()

execute_process(
  COMMAND "${EVO2C_BINARY}" --help
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error)
if(NOT help_result EQUAL 0 OR NOT help_output MATCHES "Usage:")
  message(FATAL_ERROR "evo2c --help contract failed: ${help_error}${help_output}")
endif()

execute_process(
  COMMAND "${EVO2C_BINARY}"
  RESULT_VARIABLE invalid_result
  OUTPUT_VARIABLE invalid_output
  ERROR_VARIABLE invalid_error)
if(invalid_result EQUAL 0)
  message(FATAL_ERROR "evo2c without arguments silently succeeded")
endif()
if(NOT invalid_error MATCHES "unsupported: inference is not implemented yet")
  message(FATAL_ERROR "evo2c failure was not actionable: ${invalid_error}${invalid_output}")
endif()

execute_process(
  COMMAND "${EVO2C_INSPECT_BINARY}"
  RESULT_VARIABLE inspect_result
  OUTPUT_VARIABLE inspect_output
  ERROR_VARIABLE inspect_error)
if(inspect_result EQUAL 0)
  message(FATAL_ERROR "evo2c-inspect without a model silently succeeded")
endif()
if(NOT inspect_error MATCHES "unsupported: model inspection is not implemented yet")
  message(FATAL_ERROR "evo2c-inspect failure was not actionable: ${inspect_error}${inspect_output}")
endif()

