/**
 *
 * Contains definitions for finding spots given an ImageSet
 *
 */

#ifndef IMAGESET_SPOTFINDER_H
#define IMAGESET_SPOTFINDER_H

#include <scitbx/array_family/accessors/c_grid.h>
#include <scitbx/array_family/ref.h>

namespace dials {
namespace algorithms {

class ImageSetSpotfinder {
 public:
  ImageSetSpotfinder(
      const scitbx::af::const_ref<bool, scitbx::af::c_grid<2> > &mask);
};

}  // namespace algorithms
}  // namespace dials
#endif