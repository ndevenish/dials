Python_add_library( dials_pychef_ext MODULE ext.cc )
target_link_libraries( dials_pychef_ext PUBLIC cctbx scitbx_boost_python )