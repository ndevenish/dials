project(dials)

add_tbx_module( dials INTERFACE )
target_include_directories( dials INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_libtbx_refresh_command( ${CMAKE_CURRENT_SOURCE_DIR}/libtbx_refresh.py )

add_subdirectory(algorithms)
add_subdirectory(array_family)
add_subdirectory(model)
add_subdirectory(pychef)
add_subdirectory(tests)
add_subdirectory(util)
add_subdirectory(viewer)