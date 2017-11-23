# coding: utf-8

"""
Runs tests of the Python wrapper of the C++ ImageSetSpotfinder class
"""

import pytest

from dials.algorithms.spot_finding import ImageSetSpotfinder
import scitbx.array_family.flex as flex


def test_basic_creation():
    with pytest.raises(Exception):
        i = ImageSetSpotfinder()
    basic_mask = flex.bool(10)
    ImageSetSpotfinder.configure(mask=basic_mask)
