if(NOT DEFINED EVO_BINARY)
  message(FATAL_ERROR "runtime dependency test requires EVO_BINARY")
endif()

if(APPLE)
  set(dependency_command otool -L "${EVO_BINARY}")
elseif(UNIX)
  set(dependency_command ldd "${EVO_BINARY}")
else()
  message(FATAL_ERROR "runtime dependency audit supports Linux and macOS only")
endif()

execute_process(
  COMMAND ${dependency_command}
  RESULT_VARIABLE dependency_result
  OUTPUT_VARIABLE dependency_output
  ERROR_VARIABLE dependency_error)
if(NOT dependency_result EQUAL 0)
  message(FATAL_ERROR "runtime dependency command failed: ${dependency_error}")
endif()

string(TOLOWER "${dependency_output}" dependencies_lower)
foreach(forbidden IN ITEMS torch vortex transformer_engine python)
  if(dependencies_lower MATCHES "${forbidden}")
    message(FATAL_ERROR "forbidden runtime dependency '${forbidden}': ${dependency_output}")
  endif()
endforeach()

