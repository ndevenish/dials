cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(dials)

# Add the included modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# General cmake environment configuration
include(SetDefaultBuildRelWithDebInfo) # Default builds to release with debug info
include(AlwaysColourCompilation) # Always show coloured compiler output
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Generate compile_commands.json
set(CMAKE_CXX_STANDARD 11)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(CCTBX COMPONENTS cctbx scitbx REQUIRED)
find_package(DXTBX REQUIRED)
# find_package(HDF5 REQUIRED)

# Find the boost::python library for this version of python
find_package(Boost COMPONENTS "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}" REQUIRED)

# Create Boost::python alias so we don't need to carry the python version around
if(NOT TARGET Boost::python )
    message("Adding Boost::python target")
    add_library(Boost::python INTERFACE IMPORTED)
    set_target_properties(Boost::python PROPERTIES
        INTERFACE_LINK_LIBRARIES  "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}" )
endif()

# Put the libraries into lib/ so that we can run this in-place in a TBX install
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib")

# We depend on including header sources from above this current directory.
# This currently means this _must_ be in a "dials" folder. This is unstable,
# but will be dealt with when we move to src/ layout.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

link_libraries(DXTBX::DXTBX)

# add_tbx_module( dials INTERFACE )
# target_include_directories( dials INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/.. )

# add_libtbx_refresh_command( ${CMAKE_CURRENT_SOURCE_DIR}/libtbx_refresh.py )

add_subdirectory(algorithms)
add_subdirectory(array_family)
add_subdirectory(model)
add_subdirectory(pychef)
add_subdirectory(tests)
add_subdirectory(util)
add_subdirectory(viewer)