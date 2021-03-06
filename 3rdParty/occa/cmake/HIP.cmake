find_package(HIP)

if (HIP_FOUND)
  set(WITH_HIP 1)
  set(OCCA_HIP_ENABLED 1)
  add_definitions(-D${HIP_RUNTIME_DEFINE})
  include_directories( ${HIP_INCLUDE_DIRS} )

  message(STATUS "HIP version:       ${HIP_VERSION_STRING}")
  message(STATUS "HIP platform:      ${HIP_PLATFORM}")
  message(STATUS "HIP Include Paths: ${HIP_INCLUDE_DIRS}")
  message(STATUS "HIP Libraries:     ${HIP_LIBRARIES}")
else (HIP_FOUND)
  set(WITH_HIP 0)
  set(OCCA_HIP_ENABLED 0)
endif(HIP_FOUND)
