add_python_library( dials_algorithms_spatial_indexing_ext
    boost_python/quadtree.cc
    boost_python/octree.cc
    boost_python/spatial_indexing_ext.cc )
target_link_libraries( dials_algorithms_spatial_indexing_ext scitbx_boost_python cctbx )