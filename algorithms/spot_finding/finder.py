"""
Contains implementation interface for finding spots on one or many images
"""
from __future__ import absolute_import, division, print_function

import sys
import logging
import os
import warnings
from collections import namedtuple
from concurrent.futures import as_completed
import time

from tqdm import tqdm

from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSweep
from dxtbx.model import ExperimentList

from dials.array_family import flex
from dials.model.data import PixelList, PixelListLabeller
from dials.util import Sorry, log
from dials.util.mp import BatchExecutor, MPConfig


# def do_multiple_processing(function, )

# try:
#     from typing import Tuple

#     Region = Tuple[int, int, int, int]
# except ImportError:
#     pass


logger = logging.getLogger(__name__)

Region = namedtuple("Region", ["x0", "x1", "y0", "y1"])

_no_multiprocessing_on_windows = (
    "\n"
    + "*" * 80
    + "\n"
    + "Multiprocessing is not available on windows. Setting nproc = 1, njobs = 1"
    + "\n"
    + "*" * 80
    + "\n"
)


class Result(object):
    """
    A class to hold the result from spot finding on an image.

    When doing multi processing, we can process the result of
    each thread as it comes in instead of waiting for all results.
    The purpose of this class is to allow us to set the pixel list
    to None after each image to lower memory usage.
    """

    def __init__(self, pixel_list):
        """
        Set the pixel list
        """
        self.pixel_list = pixel_list


# HACK: Record whether we're the first to read in a new process
#       This is to avoid parallel file reading of HDF5 from the same handle
_first = True


def ExtractPixelsFromImage(
    imageset,
    index,
    threshold_function,
    mask,
    region_of_interest,
    max_strong_pixel_fraction,
    compute_mean_background,
):
    """Run the spotfinding for a single image.

    Args:
        imageset: The ImageSet to load the image from
        index:    The index of the image in the imageset
        threshold_function: The configured thresholding function
        mask:       The mask to apply to the image

    Returns:
        PixelList: A list of hot pixels in this image
    """

    # # Get the frame number
    # if isinstance(imageset, ImageSweep):
    #     frame = imageset.get_array_range()[0] + index
    # else:
    #     ind = imageset.indices()
    #     if len(ind) > 1:
    #         assert all(i1 + 1 == i2 for i1, i2 in zip(ind[0:-1], ind[1:-1]))
    #     frame = ind[index]

    # Create the list of pixel lists
    # pixel_list = []


class ExtractPixelsFromImage2DNoShoeboxes(object):
    """
    A class to extract pixels from a single image
    """

    def __init__(
        self,
        imageset,
        threshold_function,
        mask,
        region_of_interest,
        max_strong_pixel_fraction,
        compute_mean_background,
        min_spot_size,
        max_spot_size,
        filter_spots,
    ):
        """
        Initialise the class

        :param imageset: The imageset to extract from
        :param threshold_function: The function to threshold with
        :param mask: The image mask
        :param region_of_interest: A region of interest to process
        :param max_strong_pixel_fraction: The maximum fraction of pixels allowed
        """
        super(ExtractPixelsFromImage2DNoShoeboxes, self).__init__(
            imageset,
            threshold_function,
            mask,
            region_of_interest,
            max_strong_pixel_fraction,
            compute_mean_background,
        )

        # Save some stuff
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots

    def __call__(self, index):
        """
        Extract strong pixels from an image

        :param index: The index of the image
        """
        # Initialise the pixel labeller
        num_panels = len(self.imageset.get_detector())
        pixel_labeller = [PixelListLabeller() for p in range(num_panels)]

        # Call the super function
        result = super(ExtractPixelsFromImage2DNoShoeboxes, self).__call__(index)

        # Add pixel lists to the labeller
        assert len(pixel_labeller) == len(result.pixel_list), "Inconsistent size"
        for plabeller, plist in zip(pixel_labeller, result.pixel_list):
            plabeller.add(plist)

        # Create shoeboxes from pixel list
        reflections, _ = pixel_list_to_reflection_table(
            self.imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=False,
        )

        # Delete the shoeboxes
        del reflections["shoeboxes"]

        # Return the reflections
        return [reflections]


class ExtractSpotsParallelTask(object):
    """
    Execute the spot finder task in parallel

    We need this external class so that we can pickle it for cluster jobs
    """

    def __init__(self, function):
        """
        Initialise with the function to call
        """
        self.function = function

    def __call__(self, task):
        """
        Call the function with th task and save the IO
        """
        log.config_simple_cached()
        result = self.function(task)
        handlers = logging.getLogger("dials").handlers
        assert len(handlers) == 1, "Invalid number of logging handlers"
        return result, handlers[0].messages()


def pixel_list_to_shoeboxes(
    imageset, pixel_labeller, min_spot_size, max_spot_size, write_hot_pixel_mask
):
    """Convert a pixel list to shoeboxes"""
    # Extract the pixel lists into a list of reflections
    shoeboxes = flex.shoebox()
    spotsizes = flex.size_t()
    hotpixels = tuple(flex.size_t() for i in range(len(imageset.get_detector())))
    if isinstance(imageset, ImageSweep):
        twod = False
    else:
        twod = True
    for i, (p, hp) in enumerate(zip(pixel_labeller, hotpixels)):
        if p.num_pixels() > 0:
            creator = flex.PixelListShoeboxCreator(
                p,
                i,  # panel
                0,  # zrange
                twod,  # twod
                min_spot_size,  # min_pixels
                max_spot_size,  # max_pixels
                write_hot_pixel_mask,
            )
            shoeboxes.extend(creator.result())
            spotsizes.extend(creator.spot_size())
            hp.extend(creator.hot_pixels())
    logger.info("")
    logger.info("Extracted {} spots".format(len(shoeboxes)))

    # Get the unallocated spots and print some info
    selection = shoeboxes.is_allocated()
    shoeboxes = shoeboxes.select(selection)
    ntoosmall = (spotsizes < min_spot_size).count(True)
    ntoolarge = (spotsizes > max_spot_size).count(True)
    assert ntoosmall + ntoolarge == selection.count(False)
    logger.info("Removed %d spots with size < %d pixels" % (ntoosmall, min_spot_size))
    logger.info("Removed %d spots with size > %d pixels" % (ntoolarge, max_spot_size))

    # Return the shoeboxes
    return shoeboxes, hotpixels


def shoeboxes_to_reflection_table(imageset, shoeboxes, filter_spots):
    """Filter shoeboxes and create reflection table"""
    # Calculate the spot centroids
    centroid = shoeboxes.centroid_valid()
    logger.info("Calculated {} spot centroids".format(len(shoeboxes)))

    # Calculate the spot intensities
    intensity = shoeboxes.summed_intensity()
    logger.info("Calculated {} spot intensities".format(len(shoeboxes)))

    # Create the observations
    observed = flex.observation(shoeboxes.panels(), centroid, intensity)

    # Filter the reflections and select only the desired spots
    flags = filter_spots(
        None, sweep=imageset, observations=observed, shoeboxes=shoeboxes
    )
    observed = observed.select(flags)
    shoeboxes = shoeboxes.select(flags)

    # Return as a reflection list
    return flex.reflection_table(observed, shoeboxes)


def pixel_list_to_reflection_table(
    imageset,
    pixel_labeller,
    filter_spots,
    min_spot_size,
    max_spot_size,
    write_hot_pixel_mask,
):
    """Convert pixel list to reflection table"""
    shoeboxes, hot_pixels = pixel_list_to_shoeboxes(
        imageset,
        pixel_labeller,
        min_spot_size=min_spot_size,
        max_spot_size=max_spot_size,
        write_hot_pixel_mask=write_hot_pixel_mask,
    )
    # Setup the reflection table converter
    return (
        shoeboxes_to_reflection_table(imageset, shoeboxes, filter_spots=filter_spots),
        hot_pixels,
    )


class ExtractSpots(object):
    """
    Class to find spots in an image and extract them into shoeboxes.
    """

    def __init__(
        self,
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction=0.1,
        compute_mean_background=False,
        mp=None,
        min_spot_size=1,
        max_spot_size=20,
        filter_spots=None,
        no_shoeboxes_2d=False,
        min_chunksize=50,
        write_hot_pixel_mask=False,
    ):
        """
        Initialise the class with the strategy

        Args:
            mp (MPConfig): The multiprocessing configuration

        :param threshold_function: The image thresholding strategy
        :param mask: The mask to use
        :param mp_method: The multi processing method
        :param nproc: The number of processors
        :param max_strong_pixel_fraction: The maximum number of strong pixels
        """
        # Set the required strategies
        self.threshold_function = threshold_function
        self.mask = mask
        # self.mp_method = mp_method
        # self.mp_chunksize = mp_chunksize
        # self.mp_nproc = mp_nproc
        # self.mp_njobs = mp_njobs
        self.mp = mp
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.region_of_interest = region_of_interest
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots
        self.no_shoeboxes_2d = no_shoeboxes_2d
        self.min_chunksize = min_chunksize
        self.write_hot_pixel_mask = write_hot_pixel_mask

    def __call__(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: reflection table of spots
        """
        if not self.no_shoeboxes_2d:
            return self._find_spots(imageset)
        else:
            return self._find_spots_2d_no_shoeboxes(imageset)

    # def _compute_chunksize(self, nimg, nproc, min_chunksize):
    #     """
    #     Compute the chunk size for a given number of images and processes
    #     """
    #     chunksize = int(math.ceil(nimg / nproc))
    #     remainder = nimg % (chunksize * nproc)
    #     test_chunksize = chunksize - 1
    #     while test_chunksize >= min_chunksize:
    #         test_remainder = nimg % (test_chunksize * nproc)
    #         if test_remainder <= remainder:
    #             chunksize = test_chunksize
    #             remainder = test_remainder
    #         test_chunksize -= 1
    #     return chunksize

    def _find_spots(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: The list of spot shoeboxes
        """
        # # Change the number of processors if necessary
        # mp_nproc = self.mp_nproc
        # mp_njobs = self.mp_njobs
        # if os.name == "nt" and (self.mp.mp_nproc > 1 or mp_njobs > 1):
        #     logger.warning(_no_multiprocessing_on_windows)
        #     mp_nproc = 1
        #     mp_njobs = 1
        # if mp_nproc * mp_njobs > len(imageset):
        #     mp_nproc = min(mp_nproc, len(imageset))
        #     mp_njobs = int(math.ceil(len(imageset) / mp_nproc))

        # mp_method = self.mp_method
        # mp_chunksize = self.mp_chunksize

        # if mp_chunksize == libtbx.Auto:
        #     mp_chunksize = self._compute_chunksize(
        #         len(imageset), mp_njobs * mp_nproc, self.min_chunksize
        #     )
        #     logger.info("Setting chunksize=%i" % mp_chunksize)

        # len_by_nproc = int(math.floor(len(imageset) / (mp_njobs * mp_nproc)))
        # if mp_chunksize > len_by_nproc:
        #     mp_chunksize = len_by_nproc
        # if mp_chunksize == 0:
        #     mp_chunksize = 1
        # assert mp_nproc > 0, "Invalid number of processors"
        # assert mp_njobs > 0, "Invalid number of jobs"
        # assert mp_njobs == 1 or mp_method is not None, "Invalid cluster method"
        # assert mp_chunksize > 0, "Invalid chunk size"

        # The extract pixels function
        function = ExtractPixelsFromImage(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
        )

        # The indices to iterate over
        indices = list(range(len(imageset)))

        # Initialise the pixel labeller
        num_panels = len(imageset.get_detector())
        pixel_labeller = [PixelListLabeller() for p in range(num_panels)]

        # Do the processing
        # logger.info("Extracting strong pixels from images")
        # if mp_njobs > 1:
        #     logger.info(
        #         " Using %s with %d parallel job(s) and %d processes per node\n"
        #         % (mp_method, mp_njobs, mp_nproc)
        #     )
        # else:
        #     logger.info(" Using multiprocessing with %d parallel job(s)\n" % (mp_nproc))
        # if mp_nproc > 1 or mp_njobs > 1:

        #     def process_output(result):
        #         for message in result[1]:
        #             logger.log(message.levelno, message.msg)
        #         assert len(pixel_labeller) == len(
        #             result[0].pixel_list
        #         ), "Inconsistent size"
        #         for plabeller, plist in zip(pixel_labeller, result[0].pixel_list):
        #             plabeller.add(plist)
        #         result[0].pixel_list = None

        #     batch_multi_node_parallel_map(
        #         func=ExtractSpotsParallelTask(function),
        #         iterable=indices,
        #         nproc=mp_nproc,
        #         njobs=mp_njobs,
        #         cluster_method=mp_method,
        #         chunksize=mp_chunksize,
        #         callback=process_output,
        #     )
        # else:
        for task in indices:
            result = function(task)
            assert len(pixel_labeller) == len(result.pixel_list), "Inconsistent size"
            for plabeller, plist in zip(pixel_labeller, result.pixel_list):
                plabeller.add(plist)
                result.pixel_list = None

        # Create shoeboxes from pixel list
        return pixel_list_to_reflection_table(
            imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=self.write_hot_pixel_mask,
        )

    def _find_spots_2d_no_shoeboxes(self, imageset):
        """
        Find the spots in the imageset

        :param imageset: The imageset to process
        :return: The list of spot shoeboxes
        """
        # # Change the number of processors if necessary
        # mp_nproc = self.mp_nproc
        # mp_njobs = self.mp_njobs
        # if os.name == "nt" and (mp_nproc > 1 or mp_njobs > 1):
        #     logger.warning(_no_multiprocessing_on_windows)
        #     mp_nproc = 1
        #     mp_njobs = 1
        # if mp_nproc * mp_njobs > len(imageset):
        #     mp_nproc = min(mp_nproc, len(imageset))
        #     mp_njobs = int(math.ceil(len(imageset) / mp_nproc))

        # mp_method = self.mp_method
        # mp_chunksize = self.mp_chunksize

        # if mp_chunksize == libtbx.Auto:
        #     mp_chunksize = self._compute_chunksize(
        #         len(imageset), mp_njobs * mp_nproc, self.min_chunksize
        #     )
        #     logger.info("Setting chunksize=%i" % mp_chunksize)

        # len_by_nproc = int(math.floor(len(imageset) / (mp_njobs * mp_nproc)))
        # if mp_chunksize > len_by_nproc:
        #     mp_chunksize = len_by_nproc
        # assert mp_nproc > 0, "Invalid number of processors"
        # assert mp_njobs > 0, "Invalid number of jobs"
        # # assert mp_njobs == 1 or mp_method is not None, "Invalid cluster method"
        # assert mp_chunksize > 0, "Invalid chunk size"

        # The extract pixels function
        function = ExtractPixelsFromImage2DNoShoeboxes(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
        )

        # The indices to iterate over
        indices = list(range(len(imageset)))

        # The resulting reflections
        reflections = flex.reflection_table()

        # Do the processing
        logger.info("Extracting strong spots from images")
        if mp_njobs > 1:
            logger.info(
                " Using %s with %d parallel job(s) and %d processes per node\n"
                % (mp_method, mp_njobs, mp_nproc)
            )
        else:
            logger.info(" Using multiprocessing with %d parallel job(s)\n" % (mp_nproc))
        if mp_nproc > 1 or mp_njobs > 1:

            def process_output(result):
                for message in result[1]:
                    logger.log(message.levelno, message.msg)
                reflections.extend(result[0][0])
                result[0][0] = None

            batch_multi_node_parallel_map(
                func=ExtractSpotsParallelTask(function),
                iterable=indices,
                nproc=mp_nproc,
                njobs=mp_njobs,
                cluster_method=mp_method,
                chunksize=mp_chunksize,
                callback=process_output,
            )
        else:
            for task in indices:
                reflections.extend(function(task)[0])

        # Return the reflections
        return reflections, None


def _placeholder_do_spotfinding(*args):
    print("Running", args)
    time.sleep(0.1)
    result = flex.reflection_table()
    #
    return result


def _apply_region_of_interest(region, *args):
    # type: (Region, Image) -> List[Image, Image]
    """
    Applies a region of interest to an image and mask.

    Args:
        region: The region to apply. If None, not transform is applied.
        *args:  Number of images to apply the region to
    Returns:
        A list of input images, restricted to the region specified. If the
        region is None, then no transform is applied.
    """
    if region is None:
        return list(args)
    assert region.x0 < region.x1, "x0 < x1"
    assert region.y0 < region.y1, "y0 < y1"
    assert region.x0 >= 0, "x0 >= 0"
    assert region.y0 >= 0, "y0 >= 0"

    images = []
    for image in args:
        height, width = image.all()
        assert region.x1 <= width, "x1 <= width"
        assert region.y1 <= height, "y1 <= height"
        im_roi = image[region.y0 : region.y1, region.x0 : region.x1]
        images.append(im_roi)

    return images


def find_spots(threshold_function, image, mask=None, region_of_interest=None):
    # type: (Callable, Image, Image, Region) -> List[PixelList]
    """Do spotfinding for a single image.

    Runs a thresholding function over an image, with a mask and region of
    interest applied.

    Args:
        threshold_function: The threshold function object to do the calculation

    Returns:
        A list of pixel list objects, one for each panel. The frame is
        set to 0 so should be changed before e.g. combining with other
        pixel lists to create shoeboxes.
    """
    # If no mask, don't mask anything
    mask = mask or flex.bool(image.accessor(), True)

    pixel_lists = []

    # Work on images/masks on a per-panel basis
    for panel_image, panel_mask in zip(image, mask):
        # Only work with a subset of the image if requested
        panel_image_roi, panel_mask_roi = _apply_region_of_interest(
            region_of_interest, panel_image, panel_mask
        )

        # Do the actual threshold mask computation
        threshold_mask_roi = threshold_function.compute_threshold(
            panel_image_roi, panel_mask_roi
        )

        # Unshift the threshold mask by region of interest if we shifted it
        if region_of_interest:
            threshold_mask = flex.bool(panel_image.accessor(), False)
            threshold_mask[
                region_of_interest.y0 : region_of_interest.y1,
                region_of_interest.x0 : region_of_interest.x1,
            ] = threshold_mask_roi
        else:
            threshold_mask = threshold_mask_roi

        # Generate a pixel list from this thresholded mask
        pixel_list = PixelList(0, panel_image, threshold_mask)
        pixel_lists.append(pixel_list)
        breakpoint()
    ###########################################################################
    # START OF EXTRACTSPOTS

    # Get the image and mask

    # Add the images to the pixel lists
    # num_strong = 0
    # average_background = 0

    # # Add the pixel list
    # plist = PixelList(frame, im, threshold_mask)
    # pixel_list.append(plist)

    # TODO: Put compute_mean_background back in
    # # # Get average background
    # # if self.compute_mean_background:
    # #     background = im.as_1d().select((mk & ~threshold_mask).as_1d())
    # #     average_background += flex.mean(background)

    # # Add to the spot count
    # num_strong += len(plist)

    # Make average background
    # average_background /= len(image)

    # TODO: Put max_strong_pixel_fraction back in
    # # Check total number of strong pixels
    # if self.max_strong_pixel_fraction < 1:
    #     num_image = 0
    #     for im in image:
    #         num_image += len(im)
    #     max_strong = int(math.ceil(self.max_strong_pixel_fraction * num_image))
    #     if num_strong > max_strong:
    #         raise RuntimeError(
    #             """
    #     The number of strong pixels found (%d) is greater than the
    #     maximum allowed (%d). Try changing spot finding parameters
    # """
    #             % (num_strong, max_strong)
    #         )

    # # Print some info
    # if self.compute_mean_background:
    #     logger.info(
    #         "Found %d strong pixels on image %d with average background %f"
    #         % (num_strong, frame + 1, average_background)
    #     )
    # else:
    #     logger.info("Found %d strong pixels on image %d" % (num_strong, frame + 1))

    # Return the result
    # return Result(pixel_list)

    # END OF EXTRACTSPOTS
    ###############################################################################
    return pixel_lists


def _build_mask(imageset, image_index, global_mask=None, mask_generator=None):
    """Build/extract/combine a mask for an image"""
    mask = None
    # Combine the global mask with a per-imageset mask
    if mask_generator:
        generated_mask = mask_generator.generate(imageset)
        if global_mask is not None:
            mask = tuple(m1 & m2 for m1, m2 in zip(global_mask, generated_mask))

    # Validate the mask size matches the detector panel structure
    if mask is not None:
        assert len(mask) == len(imageset.get_detector())

    image_mask = imageset.get_mask(image_index)

    # Set the mask
    if mask is None:
        mask = image_mask
    else:
        assert len(mask) == len(image_mask)
        mask = tuple(m1 & m2 for m1, m2 in zip(mask, image_mask))

    return mask


def _do_spotfinding(
    experiment_index,
    imageset,
    image_index,
    threshold_function,
    global_mask=None,
    mask_generator=None,
    region_of_interest=None,
    max_strong_pixel_fraction=0.1,
    compute_mean_background=False,
):
    # type: (int, ImageSet, int, Image, Region, float, bool) -> flex.reflection_table
    """
    Do the spotfinding for a single image in an imageset.

    This is called by the parallel methods
    """
    # print("Finding {}[{}]".format(experiment_index, image_index))
    time.sleep(0.1)
    # HACK: Must be a better way of doing this?
    # Parallel reading of HDF5 from the same handle is not allowed. Python
    # multiprocessing is a bit messed up and used fork on linux so need to
    # close and reopen file.
    global _first
    if _first:
        if imageset.reader().is_single_file_reader():
            imageset.reader().nullify_format_instance()
        _first = False

    # Build the mask from the component parts
    mask = _build_mask(imageset, image_index, global_mask, mask_generator)

    logger.debug(
        "Number of masked pixels for image {}[{}]: {}".format(
            imageset, image_index, sum(m.count(False) for m in mask)
        )
    )

    # Read the actual image
    image = imageset.get_corrected_data(image_index)

    pixel_lists = find_spots(
        threshold_function, image, mask=mask, region_of_interest=region_of_interest
    )

    # result = flex.reflection_table()
    # result["id"] = flex.int(result.nrows(), experiment_index)
    # return result
    return flex.reflection_table()
    return pixel_lists


class SpotFinder(object):
    """
    A class to do spot finding and filtering.
    """

    def __init__(
        self,
        # Inner per-image params
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction=0.1,
        compute_mean_background=False,
        # Middle per-imageset
        filter_spots=None,
        write_hot_mask=True,
        no_shoeboxes_2d=False,
        mp=None,
        # Params for this level
        mask_generator=None,
        scan_range=None,
        hot_mask_prefix="hot_mask",
        min_spot_size=1,
        max_spot_size=20,
    ):
        """
        Initialise the class.

        :param find_spots: The spot finding algorithm
        :param filter_spots: The spot filtering algorithm
        :param scan_range: The scan range to find spots over
        """

        # Set the filter and some other stuff
        self.threshold_function = threshold_function
        self.mask = mask
        self.region_of_interest = (
            Region(*region_of_interest) if region_of_interest else None
        )
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.mask_generator = mask_generator
        self.filter_spots = filter_spots
        self.scan_range = scan_range
        self.write_hot_mask = write_hot_mask
        self.hot_mask_prefix = hot_mask_prefix
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.mp = mp or MPConfig(
            method=None, nproc=1, njobs=1, chunksize=None, min_chunksize=None
        )

        # self.mp_method = mp_method
        # self.mp_chunksize = mp_chunksize
        # self.mp_nproc = mp_nproc
        # self.mp_njobs = mp_njobs
        self.no_shoeboxes_2d = no_shoeboxes_2d
        # self.min_chunksize = min_chunksize

    def _build_image_processing_list(self, experiments):
        """Build a list of images to process along with required data"""
        # Let's build a list of every image to find spots in
        images_to_search = []
        ImageEntry = namedtuple("ImageEntry", ["experiment_index", "imageset", "index"])
        for i, imageset in enumerate(experiments.imagesets()):
            images_to_search.extend(
                ImageEntry(i, imageset, index) for index in range(len(imageset))
            )
        return images_to_search

    def find_spots(self, experiments):
        # type: (ExperimentList) -> flex.reflection_table
        """
        Do spotfinding for a set of experiments.

        Args:
            experiments: The experiment list to process

        Returns:
            A new reflection table of found reflections
        """

        images_to_search = self._build_image_processing_list(experiments)
        logger.info(
            "Finding strong spots in {} imagesets".format(len(images_to_search))
        )

        # Each experiment should have a reflection table; merge afterwards
        # reflections = [flex.reflection_table() for _ in range(len(experiments))]
        reflections = flex.reflection_table()

        with tqdm(total=len(images_to_search), leave=False, smoothing=0) as progress:
            try:
                pool = BatchExecutor(
                    method="serial",  # self.mp.method,
                    max_workers=self.mp.nproc,
                    njobs=self.mp.njobs,
                    chunksize=self.mp.chunksize,
                )
                # Submit all the jobs
                # futures = []
                # for task in images_to_search:
                #     futures.append()

                # args = []
                #                             _do_spotfinding,
                #         task.experiment_index,
                #         task.imageset,
                #         task.index,
                #         threshold_function=self.threshold_function,
                #         global_mask=self.mask,
                #         mask_generator=self.mask_generator,
                #         max_strong_pixel_fraction=self.max_strong_pixel_fraction,
                #         compute_mean_background=self.compute_mean_background,
                # ]
                start = time.time()
                futures = []
                for task in images_to_search:
                    futures.append(
                        pool.submit(
                            _do_spotfinding,
                            task.experiment_index,
                            task.imageset,
                            task.index,
                            threshold_function=self.threshold_function,
                            global_mask=self.mask,
                            mask_generator=self.mask_generator,
                            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
                            compute_mean_background=self.compute_mean_background,
                        )
                    )

                # futures = [

                #     for task in images_to_search
                # ]
                progress.write("Submission took {:.1f}s".format(time.time() - start))
                # sys.exit("OH")
                for future in as_completed(futures):
                    progress.update(1)
                    # breakpoint()
                    reflections.extend(future.result())
                    # result = future.result()
            finally:
                # Cancel anything still being waited on
                for future in futures:
                    future.cancel()
                pool.shutdown()
        breakpoint()
        # # We now need to dispatch to multiprocessing here; for now, do linearly
        # for image in tqdm(images_to_search):
        # Do the actual spotfinding
        # table, hot_mask = self._find_spots_in_imageset(imageset)
        # TODO: Something with the hot mask

        # Assign the experiment index to all entries in this table
        # table["id"] = flex.int(table.nrows(), image.experiment_index)
        # Since reflections have correct imageset index and experiment now,
        # is safe to recombine in arbitrary order
        # reflections.extend(table)

        # See if we've requested writing a hot pixel mask
        if self.write_hot_mask:
            raise NotImplementedError(
                "Currently disabled hot mask until architecture working"
            )
            # if not imageset.external_lookup.mask.data.empty():
            #     for m1, m2 in zip(hot_mask, imageset.external_lookup.mask.data):
            #         m1 &= m2.data()
            #     imageset.external_lookup.mask.data = ImageBool(hot_mask)
            # else:
            #     imageset.external_lookup.mask.data = ImageBool(hot_mask)
            # imageset.external_lookup.mask.filename = "%s_%d.pickle" % (
            #     self.hot_mask_prefix,
            #     i,
            # )
            # # Write the hot mask
            # with open(imageset.external_lookup.mask.filename, "wb") as outfile:
            #     pickle.dump(hot_mask, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        # Set the strong spot flag to everything found in this process
        reflections.set_flags(
            flex.size_t_range(len(reflections)), reflections.flags.strong
        )

        # Update the reflection table overload flag
        reflections.is_overloaded(experiments)

        # Return the reflections
        return reflections

    def _find_spots_in_imageset(self, imageset):
        """
        Do the spot finding.

        :param imageset: The imageset to process
        :return: The observed spots
        """
        # The input mask
        mask = self.mask_generator.generate(imageset)
        if self.mask is not None:
            mask = tuple(m1 & m2 for m1, m2 in zip(mask, self.mask))

        # Set the spot finding algorithm
        extract_spots = ExtractSpots(
            threshold_function=self.threshold_function,
            mask=mask,
            region_of_interest=self.region_of_interest,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            mp=self.mp,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
            no_shoeboxes_2d=self.no_shoeboxes_2d,
            min_chunksize=self.min_chunksize,
            write_hot_pixel_mask=self.write_hot_mask,
        )

        # Get the max scan range
        if isinstance(imageset, ImageSweep):
            max_scan_range = imageset.get_array_range()
        else:
            max_scan_range = (0, len(imageset))

        # Get list of scan ranges
        if not self.scan_range or self.scan_range[0] is None:
            scan_range = [(max_scan_range[0] + 1, max_scan_range[1])]
        else:
            scan_range = self.scan_range

        # Get spots from bits of scan
        hot_pixels = tuple(flex.size_t() for i in range(len(imageset.get_detector())))
        reflections = flex.reflection_table()
        for j0, j1 in scan_range:
            # Make sure we were asked to do something sensible
            if j1 < j0:
                raise Sorry("Scan range must be in ascending order")
            elif j0 < max_scan_range[0] or j1 > max_scan_range[1]:
                raise Sorry(
                    "Scan range must be within image range {}..{}".format(
                        max_scan_range[0] + 1, max_scan_range[1]
                    )
                )

            logger.info("\nFinding spots in image {} to {}...".format(j0, j1))
            j0 -= 1
            if isinstance(imageset, ImageSweep):
                j0 -= imageset.get_array_range()[0]
                j1 -= imageset.get_array_range()[0]
            r, h = extract_spots(imageset[j0:j1])
            reflections.extend(r)
            if h is not None:
                for h1, h2 in zip(hot_pixels, h):
                    h1.extend(h2)

        # Find hot pixels
        hot_mask = self._create_hot_mask(imageset, hot_pixels)

        # Return as a reflection list
        return reflections, hot_mask

    def _create_hot_mask(self, imageset, hot_pixels):
        """
        Find hot pixels in images
        """
        # Write the hot mask
        if self.write_hot_mask:

            # Create the hot pixel mask
            hot_mask = tuple(
                flex.bool(flex.grid(p.get_image_size()[::-1]), True)
                for p in imageset.get_detector()
            )
            num_hot = 0
            if hot_pixels:
                for hp, hm in zip(hot_pixels, hot_mask):
                    for i in range(len(hp)):
                        hm[hp[i]] = False
                    num_hot += len(hp)
            logger.info("Found %d possible hot pixel(s)" % num_hot)

        else:
            hot_mask = None

        # Return the hot mask
        return hot_mask

    def __call__(self, experiments):
        warnings.warn(
            "Please use Spotfinder.find_spots to run spotfinding.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_spots(experiments)
