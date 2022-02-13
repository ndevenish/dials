add_library( shared_outlier_helpers OBJECT boost_python/outlier_helpers.cc )
target_link_libraries( shared_outlier_helpers PUBLIC Boost::python Boost::boost CCTBX::cctbx Boost::python )

Python_add_library( dials_refinement_helpers_ext MODULE
    boost_python/parameterisation_helpers.cc
    boost_python/gallego_yezzi.cc
    boost_python/mahalanobis.cc
    boost_python/restraints_helpers.cc
    boost_python/rtmats.cc
    boost_python/gaussian_smoother.cc
    boost_python/gaussian_smoother_2D.cc
    boost_python/gaussian_smoother_3D.cc
    boost_python/refinement_ext.cc
    $<TARGET_OBJECTS:shared_outlier_helpers> )
target_link_libraries( dials_refinement_helpers_ext PUBLIC shared_outlier_helpers CCTBX::cctbx Boost::python )