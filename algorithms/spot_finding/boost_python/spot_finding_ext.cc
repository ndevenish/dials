/*
 * spot_finding_ext.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <vector>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/spot_finding/helpers.h>
#include <dials/algorithms/spot_finding/imageset_spotfinder.h>


namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  /** Construct an ImageSetSpotfinder from a python list of masks.
   *
   *  This is a convenience constructor to help keep the C++ and python
   *  interfaces separate. Currently requires that the passed in object
   *  is a list-like.
   *
   */
  boost::shared_ptr<ImageSetSpotfinder> make_imagesetspotfinder_from_masks(
      const object& py_masks) {
    // Convert the input python list to a list of masks
    std::vector<scitbx::af::const_ref<bool, scitbx::af::c_grid<2>>> masks;

    for (int i = 0; i < len(py_masks); ++i) {
      masks.push_back(
          boost::python::extract<
              scitbx::af::const_ref<bool, scitbx::af::c_grid<2>>>(py_masks[i]));
    }
    // Return the new ImageSpotfinder object
    return boost::make_shared<ImageSetSpotfinder>(masks);
  }

  BOOST_PYTHON_MODULE(dials_algorithms_spot_finding_ext)
  {
    class_<StrongSpotCombiner>("StrongSpotCombiner")
      .def("add", &StrongSpotCombiner::add)
      .def("shoeboxes", &StrongSpotCombiner::shoeboxes)
      ;

    class_<ImageSetSpotfinder>("ImageSetSpotfinder", no_init)
        .def("__init__",
             make_constructor(&make_imagesetspotfinder_from_masks,
                              default_call_policies(), arg("masks")));
  }

}}}
