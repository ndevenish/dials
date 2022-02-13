Python_add_library( dials_util_ext MODULE boost_python/ext.cc )
target_link_libraries( dials_util_ext PUBLIC ccp4io Boost::python CCTBX::cctbx )

Python_add_library( dials_util_streambuf_test_ext MODULE
    boost_python/streambuf_test_ext.cpp )
target_link_libraries( dials_util_streambuf_test_ext PUBLIC ccp4io Boost::python CCTBX::cctbx )