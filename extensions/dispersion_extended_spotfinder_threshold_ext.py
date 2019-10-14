from __future__ import absolute_import, division, print_function

import logging

import libtbx
from scitbx.array_family import flex
from scitbx import matrix
from dials.algorithms.spot_finding.threshold import DispersionExtendedThresholdStrategy
from dials.util.phil import ScopedPhilScope

logger = logging.getLogger(
    "dials.extensions.dispersion_extended_spotfinder_threshold_ext"
)


class DispersionExtendedSpotFinderThresholdExt(object):
    """ Extensions to do dispersion threshold. """

    name = "dispersion_extended"

    default = True

    @classmethod
    def phil(cls):
        return None

    def __init__(self, params):
        """
        Initialise the algorithm.

        :param params: The input parameters
        """
        self.params = ScopedPhilScope(params.spotfinder)

    def compute_threshold(self, image, mask):
        """
        Compute the threshold.

        :param image: The image to process
        :param mask: The pixel mask on the image
        :returns: A boolean mask showing foreground/background pixels
        """

        if self.params.threshold.dispersion.global_threshold is libtbx.Auto:
            self.params.threshold.dispersion.global_threshold = int(
                estimate_global_threshold(image, mask)
            )
            logger.info(
                "Setting global_threshold: %i"
                % (self.params.threshold.dispersion.global_threshold)
            )

        self._algorithm = DispersionExtendedThresholdStrategy(
            kernel_size=self.params.threshold.dispersion.kernel_size,
            gain=self.params.threshold.dispersion.gain,
            mask=self.params.lookup.mask,
            n_sigma_b=self.params.threshold.dispersion.sigma_background,
            n_sigma_s=self.params.threshold.dispersion.sigma_strong,
            min_count=self.params.threshold.dispersion.min_local,
            global_threshold=self.params.threshold.dispersion.global_threshold,
        )

        return self._algorithm(image, mask)


def estimate_global_threshold(image, mask=None, plot=False):
    n_above_threshold = flex.size_t()
    threshold = flex.double()
    for i in range(1, 20):
        g = 1.5 ** i
        g = int(g)
        n_above_threshold.append((image > g).count(True))
        threshold.append(g)

    # Find the elbow point of the curve, in the same manner as that used by
    # distl spotfinder for resolution method 1 (Zhang et al 2006).
    # See also dials/algorithms/spot_finding/per_image_analysis.py

    x = threshold.as_double()
    y = n_above_threshold.as_double()
    slopes = (y[-1] - y[:-1]) / (x[-1] - x[:-1])
    p_m = flex.min_index(slopes)

    x1 = matrix.col((x[p_m], y[p_m]))
    x2 = matrix.col((x[-1], y[-1]))

    gaps = flex.double()
    v = matrix.col(((x2[1] - x1[1]), -(x2[0] - x1[0]))).normalize()

    for i in range(p_m, len(x)):
        x0 = matrix.col((x[i], y[i]))
        r = x1 - x0
        g = abs(v.dot(r))
        gaps.append(g)

    p_g = flex.max_index(gaps)

    x_g_ = x[p_g + p_m]
    y_g_ = y[p_g + p_m]

    # more conservative, choose point 2 left of the elbow point
    x_g = x[p_g + p_m - 2]
    # y_g = y[p_g + p_m - 2]

    if plot:
        from matplotlib import pyplot

        pyplot.figure(figsize=(16, 12))
        pyplot.scatter(threshold, n_above_threshold, marker="+")
        # for i in range(len(threshold)-1):
        #  pyplot.plot([threshold[i], threshold[-1]],
        #              [n_above_threshold[i], n_above_threshold[-1]])
        # for i in range(1, len(threshold)):
        #  pyplot.plot([threshold[0], threshold[i]],
        #              [n_above_threshold[0], n_above_threshold[i]])
        pyplot.plot([x_g, x_g], pyplot.ylim())
        pyplot.plot(
            [threshold[p_m], threshold[-1]],
            [n_above_threshold[p_m], n_above_threshold[-1]],
        )
        pyplot.plot([x_g_, threshold[-1]], [y_g_, n_above_threshold[-1]])
        pyplot.xlabel("Threshold")
        pyplot.ylabel("Number of pixels above threshold")
        pyplot.savefig("global_threshold.png")
        pyplot.clf()

    return x_g
