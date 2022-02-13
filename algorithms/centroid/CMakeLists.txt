add_python_library( dials_algorithms_centroid_simple_ext
    simple/boost_python/simple_ext.cc )
target_link_libraries( dials_algorithms_centroid_simple_ext PUBLIC cctbx scitbx_boost_python )