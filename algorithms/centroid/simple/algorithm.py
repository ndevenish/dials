from __future__ import absolute_import, division, print_function

from dials.algorithms.centroid.simple import Centroider


class Algorithm(object):
    """
    A python class to wrap the centroid algorithm
    """

    def __init__(self, experiments):
        """
        Initialize the centroider

        :param experiments: The experiment list
        """
        # Create the centroider
        self.centroider = Centroider()

        # Add all the experiments
        for exp in experiments:
            if exp.scan is not None:
                self.centroider.add(exp.detector, exp.scan)
            else:
                self.centroider.add(exp.detector)

    def __call__(self, reflections, image_volume=None):
        """
        Do the centroiding

        :param reflections: The reflection list
        """
        if image_volume is None:
            return self.centroider(reflections)
        return self.centroider(reflections, image_volume)
