Python_add_library( dials_algorithms_polygon_ext MODULE boost_python/polygon_ext.cc )
target_link_libraries( dials_algorithms_polygon_ext PUBLIC CCTBX::cctbx scitbx_boost_python )

Python_add_library( dials_algorithms_polygon_spatial_interpolation_ext MODULE
    boost_python/spatial_interpolation_ext.cc )
target_link_libraries( dials_algorithms_polygon_spatial_interpolation_ext PUBLIC CCTBX::cctbx scitbx_boost_python )

add_subdirectory(clip)