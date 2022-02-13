add_python_library( dials_algorithms_integration_ext
    boost_python/corrections.cc
    boost_python/integration_ext.cc )
target_link_libraries( dials_algorithms_integration_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_integration_integrator_ext
    boost_python/integration_integrator_ext.cc )
target_link_libraries( dials_algorithms_integration_integrator_ext PUBLIC cctbx scitbx_boost_python )

# dials_algorithms_integration_parallel_integrator_ext requires this normally optional dependency
if(TARGET Boost::thread)
  add_python_library( dials_algorithms_integration_parallel_integrator_ext
      boost_python/parallel_integrator_ext.cc )
  target_link_libraries( dials_algorithms_integration_parallel_integrator_ext PUBLIC Boost::thread scitbx_boost_python cctbx )
endif()

add_python_library( dials_algorithms_integration_sum_ext
    sum/boost_python/summation.cc
    sum/boost_python/integration_sum_ext.cc )
target_link_libraries( dials_algorithms_integration_sum_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_integration_fit_ext
    fit/boost_python/ext.cc )
target_link_libraries( dials_algorithms_integration_fit_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_integration_bayes_ext
    bayes/boost_python/ext.cc )
target_link_libraries( dials_algorithms_integration_bayes_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_integration_kapton_ext
    boost_python/kapton_ext.cc )
target_link_libraries( dials_algorithms_integration_kapton_ext PUBLIC cctbx scitbx_boost_python )