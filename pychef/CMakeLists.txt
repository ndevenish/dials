add_python_library( dials_pychef_ext ext.cc )
target_link_libraries( dials_pychef_ext PUBLIC cctbx scitbx_boost_python )