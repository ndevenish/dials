# coding: utf-8
"""
Asynchronous image loader
"""

from __future__ import absolute_import, division, print_function

# from collections import namedtuple
import logging
import numbers
import queue
import threading

# from dxtbx.imageset import ImageSweep

logger = logging.getLogger(__name__)
# Set up thread-local-logging
thread = threading.local()
thread.logger = logger


class ImagePacket(object):
    """Package of data representing all that is needed to spotfind.

    The data held by an ImagePacket represents a single, continuous
    rect of image data - after all panel, module logic has been removed.

    Wherever possible, this data is before corrections have been applied
    to allow flexibility in later processing.

    Attributes:
        data:           The 2D image data representing the detector are
        gain_map:       If present, the 2D image gain map
        gain (float):   A single gain value to apply to the image data
        mask (flex.bool): An image data mask
        imageset (ImageSet):    The imageset that this image came from
        index (int):            The image index from the imageset
        panel (int):            The panel from the image
        origin_region (Tuple[int,int,int,int]):
            If present, the (x, y, w, h) sub-region of the panel that
            this data is copied from. The actual image data object may
            be larger than this.
    """

    def __init__(self, imageset, index, panel):
        self.imageset = imageset
        self.index = index
        self.panel = panel

        self.data = None
        self.gain_map = None
        self.gain = 1.0
        self.mask = None
        self.origin_region = None


def _read_image_data(imageset, index):
    """Reads raw image data from an imageset for processing.

    Args:
        imageset (dxtbx.imageset.ImageSet): The imageset to load from
        index (int): The index of the image to load

    Returns:
        List[ImagePacket]: Image data for each panel and region
    """
    logger.debug("Reading ImageSet %s image %d", imageset, index)
    # Get the image and mask
    # image = self.imageset.get_corrected_data(index)
    # mask = self.imageset.get_mask(index)


class AsyncImageLoader(object):
    """
    Read images in preparation for spotfinding, asynchronously.

    Args:
        imagesets (Iterable[ImageSet]): The imagesets to load
        maxsize (int):  The maximum number of images to load at once

    Attributes:
        image_queue (Queue[ImagePacket])
    """

    def __init__(self, imagesets=[], maxsize=0):
        self._image_indices = queue.Queue()
        self._images = queue.Queue(maxsize)
        self._started = False

        # Enqueue separate entries for each panel part.
        for imageset in imagesets:
            self.add(imageset)

        logger.debug("Created %d image tasks in queue", self._image_indices.qsize())

        # Keep track of our threads
        self._threads = set()

    def add(self, imageset, index_range=None):
        """Add an ImageSet to the load queue.

        Cannot be used after the queue has been started because the
        queue could be empty, which at the moment terminates all the
        threads.

        Args:
            imageset (dxtbx.imageset.ImageSet): The ImageSet
            index_range (int or Tuple[int,int] or None):
                The index or range of indices to enqueue for loading. If
                None, then all images in the imageset range are added. If
                an integer, then a single image is added. If a Tuple then
                all images in range [start, end) are added. If either of
                the range entries are None, then images from the specified
                start index to the end, or images from the start to the
                specified index will be added.
        """
        assert not self._started
        # Convert image_range to an actual range min/max
        if hasattr(imageset, "get_array_range"):
            max_scan_range = imageset.get_array_range()
        else:
            max_scan_range = (0, len(imageset))
        if index_range is None:
            index_range = max_scan_range
        elif isinstance(index_range, numbers.Integral):
            index_range = (index_range, index_range + 1)
        else:
            start, end = index_range
            if start is None:
                start = max_scan_range[0]
            if end is None:
                end = max_scan_range[1]
            index_range = (start, end)
        # Enqueue the selected indices
        for index in range(*index_range):
            self._image_indices.put((imageset, index))

    @property
    def image_queue(self):
        return self._images

    def start(self):
        """Start processing threads.

        Threads will load and prepare images for processing, waiting if
        the image queue is full. Call join() to wait for all loading to
        be complete.
        """
        assert self._image_indices.qsize()
        self._started = True
        # Launch threads
        thread = threading.Thread(None, self._worker_start)
        thread.daemon = True
        thread.start()
        self._threads.add(thread)

    def join(self):
        """Wait until all image regions have been processed."""
        self._image_indices.join()
        # Make sure every thread is joined
        for thread in self._threads:
            thread.join()

    def _worker_start(self):
        """The entry point for each worker thread"""
        print("Entering thread")

        curthread = threading.current_thread()
        thread.logger = logging.getLogger(__name__ + "." + str(curthread.ident))
        try:
            while True:
                region_index = self._image_indices.get(block=False)
                thread.logger.debug("Pulled region %s for reading", region_index)
                # Read this image region
                image_data = _read_image_data(*region_index)
                # Put into the output queue and mark as done
                self._images.put(image_data)
                self._image_indices.task_done()
        except queue.Empty:
            pass
