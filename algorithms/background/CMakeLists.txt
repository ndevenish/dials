add_python_library( dials_algorithms_background_ext
    boost_python/helpers.cc
    boost_python/background_ext.cc )
target_link_libraries( dials_algorithms_background_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_background_modeller_ext
    boost_python/modeller_ext.cc )
target_link_libraries( dials_algorithms_background_modeller_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_background_simple_ext
    simple/boost_python/outlier_rejector.cc
    simple/boost_python/modeller.cc
    simple/boost_python/creator.cc
    simple/boost_python/background_simple_ext.cc )
target_link_libraries( dials_algorithms_background_simple_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_background_glm_ext
    glm/boost_python/ext.cc )
target_link_libraries( dials_algorithms_background_glm_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_background_median_ext
    median/boost_python/ext.cc )
target_link_libraries( dials_algorithms_background_median_ext PUBLIC cctbx scitbx_boost_python )

add_python_library( dials_algorithms_background_gmodel_ext
    gmodel/boost_python/ext.cc )
target_link_libraries( dials_algorithms_background_gmodel_ext PUBLIC cctbx scitbx_boost_python )