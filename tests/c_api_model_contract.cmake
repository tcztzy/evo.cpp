if(NOT DEFINED EVO_FIXTURE_WRITER OR
   NOT DEFINED EVO_C_API_BINARY OR
   NOT DEFINED EVO_FIXTURE_PATH)
  message(FATAL_ERROR "C API model contract requires writer, binary, and fixture")
endif()

get_filename_component(fixture_directory "${EVO_FIXTURE_PATH}" DIRECTORY)
file(MAKE_DIRECTORY "${fixture_directory}")

execute_process(
  COMMAND "${EVO_FIXTURE_WRITER}" --write-fixture "${EVO_FIXTURE_PATH}"
  RESULT_VARIABLE write_result
  OUTPUT_VARIABLE write_output
  ERROR_VARIABLE write_error)
if(NOT write_result EQUAL 0)
  message(FATAL_ERROR "fixture writer failed: ${write_error}${write_output}")
endif()

execute_process(
  COMMAND "${EVO_C_API_BINARY}" "${EVO_FIXTURE_PATH}"
  RESULT_VARIABLE api_result
  OUTPUT_VARIABLE api_output
  ERROR_VARIABLE api_error)
if(NOT api_result EQUAL 0)
  message(FATAL_ERROR "C API model contract failed: ${api_error}${api_output}")
endif()

file(REMOVE "${EVO_FIXTURE_PATH}")
