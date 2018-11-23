# coding: utf-8

from __future__ import absolute_import, division, print_function

from mock import Mock
import pytest

import dials.algorithms.spot_finding.imageloader as imageloader


@pytest.fixture
def no_read_data(monkeypatch):
    """Prevents reading real data with the image loader"""
    monkeypatch.setattr(imageloader, "_read_image_data", Mock())


pytestmark = pytest.mark.usefixtures("no_read_data")


def test_initial_setup():
    il = imageloader.AsyncImageLoader([])
