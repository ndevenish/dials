Python_add_library( dials_algorithms_indexing_ext MODULE
    boost_python/fft3d.cc
    boost_python/indexing_ext.cc )
target_link_libraries( dials_algorithms_indexing_ext PUBLIC ann Boost::python CCTBX::cctbx )