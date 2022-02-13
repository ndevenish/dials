add_python_library( dials_scaling_ext
    boost_python/scaling_helper.cc
    boost_python/scaling_ext.cc )
target_link_libraries( dials_scaling_ext PUBLIC cctbx scitbx_boost_python )