if(NOT DEFINED EVO_BINARY OR NOT DEFINED EVO_INSPECT_BINARY)
  message(FATAL_ERROR "CLI contract test requires both executable paths")
endif()

execute_process(
  COMMAND "${EVO_BINARY}" --version
  RESULT_VARIABLE version_result
  OUTPUT_VARIABLE version_output
  ERROR_VARIABLE version_error)
if(NOT version_result EQUAL 0 OR NOT version_output MATCHES "^evo 0\\.1\\.0")
  message(FATAL_ERROR "evo --version contract failed: ${version_error}${version_output}")
endif()

execute_process(
  COMMAND "${EVO_BINARY}" --help
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error)
if(NOT help_result EQUAL 0 OR NOT help_output MATCHES "Usage:")
  message(FATAL_ERROR "evo --help contract failed: ${help_error}${help_output}")
endif()
foreach(command IN ITEMS run score logits embed variant-score serve bench)
  if(NOT help_output MATCHES "evo ${command}")
    message(FATAL_ERROR "evo --help omitted ${command} command")
  endif()
endforeach()

execute_process(
  COMMAND "${EVO_BINARY}" bench --help
  RESULT_VARIABLE command_help_result
  OUTPUT_VARIABLE command_help_output
  ERROR_VARIABLE command_help_error)
if(NOT command_help_result EQUAL 0 OR
   NOT command_help_output MATCHES "--repetitions")
  message(FATAL_ERROR "subcommand help contract failed: ${command_help_error}${command_help_output}")
endif()

execute_process(
  COMMAND "${EVO_BINARY}"
  RESULT_VARIABLE invalid_result
  OUTPUT_VARIABLE invalid_output
  ERROR_VARIABLE invalid_error)
if(invalid_result EQUAL 0)
  message(FATAL_ERROR "evo without arguments silently succeeded")
endif()
if(NOT invalid_error MATCHES "invalid_argument: specify exactly one nonempty model source")
  message(FATAL_ERROR "evo failure was not actionable: ${invalid_error}${invalid_output}")
endif()

execute_process(
  COMMAND "${EVO_BINARY}" -m fake.safetensors -p "Aé" -n 1 --gpu 0 --dump-tokens
  RESULT_VARIABLE token_result
  OUTPUT_VARIABLE token_output
  ERROR_VARIABLE token_error)
string(FIND "${token_error}" "tokens prompt=[65,195,169]" token_position)
string(FIND "${token_error}" "unsupported: this evo binary was built without CUDA support" no_cuda_position)
string(FIND "${token_error}" "io: open 'fake.safetensors':" cuda_position)
if(token_result EQUAL 0 OR token_position EQUAL -1 OR
   (no_cuda_position EQUAL -1 AND cuda_position EQUAL -1))
  message(FATAL_ERROR "byte-token CLI contract failed: ${token_error}${token_output}")
endif()

execute_process(
  COMMAND "${EVO_INSPECT_BINARY}"
  RESULT_VARIABLE inspect_result
  OUTPUT_VARIABLE inspect_output
  ERROR_VARIABLE inspect_error)
if(inspect_result EQUAL 0)
  message(FATAL_ERROR "evo-inspect without a model silently succeeded")
endif()
if(NOT inspect_error MATCHES "invalid_argument: usage: evo-inspect")
  message(FATAL_ERROR "evo-inspect failure was not actionable: ${inspect_error}${inspect_output}")
endif()
