if(NOT DEFINED EVO_CUDA_BINARY OR NOT DEFINED EVO_CUOBJDUMP)
  message(FATAL_ERROR "FP8 SASS audit requires EVO_CUDA_BINARY and EVO_CUOBJDUMP")
endif()

execute_process(
  COMMAND "${EVO_CUOBJDUMP}" --dump-sass --dump-ptx "${EVO_CUDA_BINARY}"
  RESULT_VARIABLE audit_result
  OUTPUT_VARIABLE audit_output
  ERROR_VARIABLE audit_error)
if(NOT audit_result EQUAL 0)
  message(FATAL_ERROR "cuobjdump failed: ${audit_error}")
endif()

string(TOLOWER "${audit_output}" audit_lower)
if(NOT audit_lower MATCHES "sm_80")
  message(FATAL_ERROR "CUDA binary does not contain sm_80 code")
endif()

# Match instruction spellings, not host API or test names.  The implementation
# intentionally mentions E4M3 in its public symbol names, while sm_80 code must
# not contain native FP8 conversions or matrix instructions.
foreach(forbidden IN ITEMS
    "cvt[^\\n]*\\.e4m3"
    "cvt[^\\n]*\\.e5m2"
    "mma[^\\n]*\\.e4m3"
    "mma[^\\n]*\\.e5m2"
    "wgmma[^\\n]*\\.e4m3"
    "wgmma[^\\n]*\\.e5m2")
  if(audit_lower MATCHES "${forbidden}")
    message(FATAL_ERROR
      "CUDA binary contains forbidden hardware-FP8 marker '${forbidden}'")
  endif()
endforeach()
