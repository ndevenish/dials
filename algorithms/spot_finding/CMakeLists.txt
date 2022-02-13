add_python_library( dials_algorithms_spot_finding_ext
    boost_python/spot_finding_ext.cc )
target_link_libraries( dials_algorithms_spot_finding_ext PUBLIC cctbx scitbx_boost_python )