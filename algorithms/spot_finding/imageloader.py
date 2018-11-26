# coding: utf-8
"""
Asynchronous image loader
"""

from __future__ import absolute_import, division, print_function

# from collections import namedtuple
import logging
import queue
import threading


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


def _read_image_data(imageset, index, panel):
    pass


class AsyncImageLoader(object):
    """
    Read images in preparation for spotfinding, asynchronously.

    Args:
        imagesets (Iterable[ImageSet]): The imagesets to load
        maxsize (int):  The maximum number of images to load at once

    Attributes:
        image_queue (Queue[ImagePacket])
    """

    def __init__(self, imagesets, maxsize=0):
        self._image_indices = queue.Queue()
        self._images = queue.Queue(maxsize)
        # Enqueue separate entries for each panel part.
        for imageset in imagesets:
            for imageindex in range(len(imageset)):
                # Had throught about iterating over panels - but appears
                # to be no provision for reading separate panels?
                # for panel in range(len(imageset.get_detector())):
                self._image_indices.put((imageset, imageindex))

        logger.debug("Created %d image tasks in queue", self._image_indices.qsize())

        # Keep track of our threads
        self._threads = set()

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
        # Launch threads
        thread = threading.Thread(None, self._worker_start)
        thread.daemon = True
        thread.start()
        self._threads.add(thread)

    def join(self):
        """Wait until all image regions have been processed."""
        self._image_indices.join()

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
