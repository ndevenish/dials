# coding: utf-8

"""
Runs tests of the Python wrapper of the C++ ImageSetSpotfinder class
"""

import pytest

from dials.algorithms.spot_finding import ImageSetSpotfinder
import scitbx.array_family.flex as flex
from dxtbx.format.image import ImageDouble


def test_basic_creation():
    with pytest.raises(Exception):
        i = ImageSetSpotfinder()

    empty_masks = [flex.bool(flex.grid(10, 10))]
    ImageSetSpotfinder.configure(mask=empty_masks)
    ImageSetSpotfinder.configure(mask=tuple(empty_masks))
