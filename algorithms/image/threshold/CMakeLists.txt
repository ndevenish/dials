add_python_library( dials_algorithms_image_threshold_ext
    boost_python/unimodal.cc
    boost_python/local.cc
    boost_python/threshold_ext.cc )
target_link_libraries( dials_algorithms_image_threshold_ext PUBLIC scitbx_boost_python cctbx )

# Optional dependency on openmp
if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(dials_algorithms_image_threshold_ext PUBLIC OpenMP::OpenMP_CXX)
endif()