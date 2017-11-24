
#include <vector>

#include <gtest/gtest.h>
#include <boost/python.hpp>

#include <scitbx/array_family/shared.h>
#include <dials/algorithms/spot_finding/imageset_spotfinder.h>
#include <dxtbx/format/image.h>

using namespace dials::algorithms;

namespace af = scitbx::af;

template<typename T>
using image_ref = af::const_ref<T, af::c_grid<2>>;


TEST(ImageSetSpotfinder, basic_creation) {
  auto mask_store = af::shared<bool>(100);
  auto mask = image_ref<bool>(mask_store.begin(),
                                                 af::c_grid<2>(10, 10));
  auto masks = std::vector<image_ref<bool>>{mask};

  // auto mask = Image<double>();
  // boost::python::tuple masks = boost::python::make_tuple(mask);

  auto i = ImageSetSpotfinder(masks);
}