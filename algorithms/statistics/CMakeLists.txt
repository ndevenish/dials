add_python_library( dials_algorithms_statistics_ext
    boost_python/statistics_ext.cc )
target_link_libraries( dials_algorithms_statistics_ext PUBLIC cctbx scitbx_boost_python )