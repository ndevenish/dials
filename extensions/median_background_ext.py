from __future__ import absolute_import, division, print_function

from libtbx.phil import parse

from dials.algorithms.background.median import BackgroundAlgorithm


class MedianBackgroundExt(object):
    """ An extension class implementing a median background algorithm. """

    name = "median"

    @classmethod
    def phil(cls):
        return parse("")

    def __init__(self, params, experiments):
        """
        Initialise the algorithm.

        :param params: The input parameters
        :param experiments: The list of experiments
        """
        self._algorithm = BackgroundAlgorithm(experiments)

    def compute_background(self, reflections, image_volume=None):
        """
        Compute the background.

        :param reflections: The list of reflections
        """
        return self._algorithm.compute_background(
            reflections, image_volume=image_volume
        )
