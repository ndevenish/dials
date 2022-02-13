add_python_library( dials_util_ext boost_python/ext.cc )
target_link_libraries( dials_util_ext PUBLIC ccp4io scitbx_boost_python cctbx )

add_python_library( dials_util_streambuf_test_ext
    boost_python/streambuf_test_ext.cpp )
target_link_libraries( dials_util_streambuf_test_ext PUBLIC ccp4io scitbx_boost_python cctbx )