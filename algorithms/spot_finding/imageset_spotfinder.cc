/** imageset_spotfinder.cc
 *
 * C++ version of the spot extractor, that coordinates large spot-finding
 * jobs via a multithreaded approach.
 */

#include "imageset_spotfinder.h"

#include <iostream>

using std::cout;
using std::endl;

using namespace dials::algorithms;

ImageSetSpotfinder::ImageSetSpotfinder(
    std::vector<scitbx::af::const_ref<bool, scitbx::af::c_grid<2> > > masks) {
  cout << "In constructor with " << masks.size() << " masks" << endl;
}