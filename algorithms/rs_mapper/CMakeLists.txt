add_python_library( recviewer_ext ext.cpp )
target_link_libraries( recviewer_ext PUBLIC cctbx scitbx_boost_python )