# coding: utf-8
"""
Provides convenience functions and constructors for the C++ ImageSetSpotfinder
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import boost.python
from dials.algorithms.spot_finding import ImageSetSpotfinder

import logging

logger = logging.getLogger()


class ImageSetSpotfinder_aux(boost.python.injector, ImageSetSpotfinder):
    @staticmethod
    def configure(
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction=0.1,
        compute_mean_background=False,
        mp_method=None,
        mp_nproc=1,
        mp_njobs=1,
        mp_chunksize=1,
        min_spot_size=1,
        max_spot_size=20,
        filter_spots=None,
        no_shoeboxes_2d=False,
        min_chunksize=50,
        write_hot_pixel_mask=False,
    ):
        """
        Validates the configuration and returns an imageset spotfinder.

        Static creation method as it didn't seem initially possible to inject a
        new __init__ constructor for a boost.python class.

        :returns: A configured ImageSetSpotfinder
        """
        # Set the required strategies
        # self.threshold_function = threshold_function
        # self.mask = mask
        # self.mp_method = mp_method
        # self.mp_chunksize = mp_chunksize
        # self.mp_nproc = mp_nproc
        # self.mp_njobs = mp_njobs
        # self.max_strong_pixel_fraction = max_strong_pixel_fraction
        # self.compute_mean_background = compute_mean_background
        # # self.region_of_interest = region_of_interest
        # self.min_spot_size = min_spot_size
        # self.max_spot_size = max_spot_size
        # self.filter_spots = filter_spots
        # self.no_shoeboxes_2d = no_shoeboxes_2d
        # self.min_chunksize = min_chunksize
        # self.write_hot_pixel_mask = write_hot_pixel_mask

        # Validate the assumptions made in this verison of the class
        # For now, only support the normal thresholding function
        assert threshold_function is None or isinstance(
            threshold_function, DispersionSpotFinderThresholdExt
        ), "Unexpected threshold function"
        assert not no_shoeboxes_2d, "MT ExtractSpots does not support no_shoeboxes_2d"
        assert (
            mp_method is None or mp_method == "threads"
        ), "MT ExtractSpots called with wrong mp_method ({})".format(mp_method)
        assert region_of_interest is None

        return ImageSetSpotfinder(mask)

    def __call__(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process. May be an ImageSweep.
        :return: The list of spot shoeboxes? Whatever PixelListToReflectionTable returns.
        """

        # if not self.no_shoeboxes_2d:
        #   return self._find_spots(imageset)
        # else:
        #   return self._find_spots_2d_no_shoeboxes(imageset)

        # # Create shoeboxes from pixel list
        # converter = PixelListToReflectionTable(
        #   self.min_spot_size,
        #   self.max_spot_size,
        #   self.filter_spots,
        #   self.write_hot_pixel_mask)
        # return converter(imageset, pixel_labeller)
        pass

        # def _find_spots(self, imageset):
        """Do spotfinding on an imageset"""
