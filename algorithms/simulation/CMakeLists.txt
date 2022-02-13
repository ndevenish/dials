add_python_library( dials_algorithms_simulation_ext
    boost_python/simulation_ext.cc )
target_link_libraries( dials_algorithms_simulation_ext PUBLIC cctbx scitbx_boost_python )