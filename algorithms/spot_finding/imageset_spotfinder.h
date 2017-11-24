/**
 *
 * Contains definitions for finding spots given an ImageSet
 *
 */

#ifndef IMAGESET_SPOTFINDER_H
#define IMAGESET_SPOTFINDER_H

#include <vector>

#include <scitbx/array_family/accessors/c_grid.h>
#include <scitbx/array_family/ref.h>

namespace dials {
namespace algorithms {

/** Extract spots from an ImageSet object. */
class ImageSetSpotfinder {
 public:
  /// Construct given a vector of mask references
  ImageSetSpotfinder(
      std::vector<scitbx::af::const_ref<bool, scitbx::af::c_grid<2> > > masks);
};

}  // namespace algorithms
}  // namespace dials

#endif