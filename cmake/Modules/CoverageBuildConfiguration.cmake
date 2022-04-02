
if (CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    set(CMAKE_CXX_FLAGS_COVERAGE            "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-instr-generate -fcoverage-mapping")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(CMAKE_CXX_FLAGS_COVERAGE            "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fprofile-abs-path")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE     "${CMAKE_EXE_LINKER_FLAGS_COVERAGE} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE  "${CMAKE_SHARED_LINKER_FLAGS_COVERAGE} --coverage")
else()
    message(FATAL_ERROR "Did not know how to run Coverage build for ${CMAKE_CXX_COMPILER_ID}")
endif()

mark_as_advanced(
    CMAKE_C_FLAGS_COVERAGE
    CMAKE_CXX_FLAGS_COVERAGE
    CMAKE_EXE_LINKER_FLAGS_COVERAGE
    CMAKE_SHARED_LINKER_FLAGS_COVERAGE
)

# Add the "Coverage" build type to GUI build options
get_property(_config_list CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS)
if (NOT _config_list MATCHES "Coverage")
    list(APPEND _config_list "Coverage")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${_config_list}")
endif()
