# coding: utf-8

from __future__ import absolute_import, division, print_function

from mock import Mock

from dials.algorithms.spot_finding.finder import _combine_imageset_scan_range


def test_combine_imageset_scan_range():
    class my_imageset(list):
        """Custom 'imageset' that pretends to be an ImageSweep-like"""

        def get_array_range(self):
            return (1, len(self))

    assert _combine_imageset_scan_range([Mock(), Mock()]) == [(0, 2)]
    assert _combine_imageset_scan_range(my_imageset([Mock(), Mock(), Mock()])) == [
        (1, 3)
    ]

    long_imageset = [Mock() for _ in range(10)]
    assert _combine_imageset_scan_range(long_imageset) == [(0, 10)]
    # Scan ranges should be rewritten to range-based syntax
    assert _combine_imageset_scan_range(long_imageset, [(1, 3), (5, 8)]) == [
        (1, 4),
        (5, 9),
    ]
