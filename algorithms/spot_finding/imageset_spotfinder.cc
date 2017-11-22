/** extract_spots.cc
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
    const scitbx::af::const_ref<bool, scitbx::af::c_grid<2> > &mask) {
  cout << "In constructor" << endl;
}