add_python_library( dials_algorithms_polygon_clip_ext
    boost_python/clip_ext.cc )
target_link_libraries( dials_algorithms_polygon_clip_ext PUBLIC cctbx scitbx_boost_python )