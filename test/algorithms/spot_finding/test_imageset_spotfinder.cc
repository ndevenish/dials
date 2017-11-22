
#include <gtest/gtest.h>

#include <scitbx/array_family/shared.h>
#include <dials/algorithms/spot_finding/imageset_spotfinder.h>

using namespace dials::algorithms;
// using scitbx::arra

namespace af = scitbx::af;

TEST(ImageSetSpotfinder, basic_creation) {
  auto mask_store = af::shared<bool>(100);
  auto mask = af::const_ref<bool, af::c_grid<2>>(mask_store.begin(),
                                                 af::c_grid<2>(10, 10));
  auto i = ImageSetSpotfinder(mask);
}