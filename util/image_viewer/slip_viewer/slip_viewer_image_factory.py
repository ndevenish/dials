from __future__ import absolute_import, division, print_function

import iotbx.detectors

from dxtbx.format.FormatPYunspecified import FormatPYunspecified

# store default ImageFactory function
defaultImageFactory = iotbx.detectors.ImageFactory


def SlipViewerImageFactory(filename):
    try:
        return NpyImageFactory(filename)
    except Exception:
        return defaultImageFactory(filename)


# Use the dxtbx class as it handles all possible variance of NPY images
def NpyImageFactory(filename):
    img = FormatPYunspecified(filename)
    return img.get_detectorbase()
