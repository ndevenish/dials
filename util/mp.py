from __future__ import absolute_import, division, print_function

import ctypes
import math
import os
import threading
import warnings
from collections import namedtuple
from concurrent.futures import Executor, Future

import future.moves.itertools as itertools
import psutil

import libtbx.easy_mp

from dials.util.cluster_map import cluster_map as drmaa_parallel_map


def available_cores() -> int:
    """
    Determine the number of available processor cores.

    There are a number of different methods to get this information, some of
    which may not be available on a specific OS and/or version of Python. So try
    them in order and return the first successful one.
    """

    nproc = os.environ.get("NSLOTS", 0)
    try:
        nproc = int(nproc)
        if nproc >= 1:
            return nproc
    except ValueError:
        pass

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    try:
        return len(psutil.Process().cpu_affinity())
    except AttributeError:
        pass

    nproc = os.cpu_count()
    if nproc is not None:
        return nproc

    nproc = psutil.cpu_count()
    if nproc is not None:
        return nproc

    return 1


def parallel_map(
    func,
    iterable,
    processes=1,
    nslots=1,
    method=None,
    asynchronous=True,
    callback=None,
    preserve_order=True,
    preserve_exception_message=True,
    job_category="low",
):
    """
    A wrapper function to call either drmaa or easy_mp to do a parallel map
    calculation. This function is setup so that in each case we can select
    the number of cores on a machine
    """
    if method == "drmaa":
        return drmaa_parallel_map(
            func=func,
            iterable=iterable,
            callback=callback,
            nslots=nslots,
            njobs=processes,
            job_category=job_category,
        )
    else:
        qsub_command = "qsub -pe smp %d" % nslots
        return libtbx.easy_mp.parallel_map(
            func=func,
            iterable=iterable,
            callback=callback,
            method=method,
            processes=processes,
            qsub_command=qsub_command,
            asynchronous=asynchronous,
            preserve_order=preserve_order,
            preserve_exception_message=preserve_exception_message,
        )


class MultiNodeClusterFunction(object):
    """
    A function called by the multi node parallel map. On each cluster node, a
    nested parallel map using the multi processing method will be used.
    """

    def __init__(
        self,
        func,
        nproc=1,
        asynchronous=True,
        preserve_order=True,
        preserve_exception_message=True,
    ):
        """
        Init the function
        """
        self.func = func
        self.nproc = nproc
        self.asynchronous = asynchronous
        self.preserve_order = (preserve_order,)
        self.preserve_exception_message = preserve_exception_message

    def __call__(self, iterable):
        """
        Call the function
        """
        return libtbx.easy_mp.parallel_map(
            func=self.func,
            iterable=iterable,
            processes=self.nproc,
            method="multiprocessing",
            asynchronous=self.asynchronous,
            preserve_order=self.preserve_order,
            preserve_exception_message=self.preserve_exception_message,
        )


def _iterable_grouper(iterable, chunk_size):
    """
    Group the iterable into chunks of up to chunk_size items
    """
    args = [iter(iterable)] * chunk_size
    for group in itertools.zip_longest(*args):
        group = tuple(item for item in group if item is not None)
        yield group


def _create_iterable_wrapper(function):
    """
    Wraps a function so that it takes iterables and when called is applied to
    each element of the iterable and returns a list of the return values.
    """

    def run_function(iterable):
        return [function(item) for item in iterable]

    return run_function


def multi_node_parallel_map(
    func,
    iterable,
    njobs=1,
    nproc=1,
    cluster_method=None,
    asynchronous=True,
    callback=None,
    preserve_order=True,
    preserve_exception_message=True,
):
    """
    A wrapper function to call a function using multiple cluster nodes and with
    multiple processors on each node
    """

    # The function to all on the cluster
    cluster_func = MultiNodeClusterFunction(
        func=func,
        nproc=nproc,
        asynchronous=asynchronous,
        preserve_order=preserve_order,
        preserve_exception_message=preserve_exception_message,
    )

    # Create the cluster iterable
    cluster_iterable = _iterable_grouper(iterable, nproc)

    # Create the cluster callback
    if callback is not None:
        cluster_callback = _create_iterable_wrapper(callback)
    else:
        cluster_callback = None

    # Do the parallel map on the cluster
    result = parallel_map(
        func=cluster_func,
        iterable=cluster_iterable,
        callback=cluster_callback,
        method=cluster_method,
        nslots=nproc,
        processes=njobs,
        asynchronous=asynchronous,
        preserve_order=preserve_order,
        preserve_exception_message=preserve_exception_message,
    )

    # return result
    return [item for rlist in result for item in rlist]


def batch_multi_node_parallel_map(
    func=None,
    iterable=None,
    nproc=1,
    njobs=1,
    callback=None,
    cluster_method=None,
    chunksize=1,
):
    """
    A function to run jobs in batches in each process
    """
    # Call the batches in parallel
    return multi_node_parallel_map(
        func=_create_iterable_wrapper(func),
        iterable=_iterable_grouper(iterable, chunksize),
        nproc=nproc,
        njobs=njobs,
        cluster_method=cluster_method,
        callback=_create_iterable_wrapper(callback),
        preserve_order=True,
        preserve_exception_message=True,
    )


class _ParallelTask(object):
    """Allows a function to be called and wraps indexing information.

    Does you monnad?
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, args):
        index, item = args
        result = self.function(item)
        return index, result


MPConfig = namedtuple(
    "MPConfig", ["method", "njobs", "nproc", "chunksize", "min_chunksize"]
)


def _compute_chunksize(nimg, nproc, min_chunksize):
    """
    Compute the chunk size for a given number of images and processes

    Args:
        nimg: The number of images
        nproc: The number of processes
        min_chunksize: The minimum chunksize
    """
    chunksize = int(math.ceil(nimg / nproc))
    remainder = nimg % (chunksize * nproc)
    test_chunksize = chunksize - 1
    while test_chunksize >= min_chunksize:
        test_remainder = nimg % (test_chunksize * nproc)
        if test_remainder <= remainder:
            chunksize = test_chunksize
            remainder = test_remainder
        test_chunksize -= 1
    return chunksize


# HACK: Need to verify operation/provenance of this function
def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.isAlive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class BatchExecutor(Executor):
    """Wrapper to abstract away dealing with easy_mp and multiprocessing."""

    def __init__(
        self, method, nprocs, njobs=1, default_chunksize=None, min_chunksize=20
    ):
        if default_chunksize is libtbx.Auto:
            default_chunksize = None
        chunksize = default_chunksize
        self.config = MPConfig(method, njobs, nprocs, chunksize, min_chunksize)
        self._threads = []

    def submit(self, func, *args, **kwargs):
        raise NotImplementedError("Individual function submission not implemented")

    def _threaded_do_parallel_map_call(self, chunksize, func, futures):
        """Called in a thread. Calls the parallel map function.

        Args:
            chunksize: Individual submissions can override the configuration
            func: The function to call
            futures: The Future objects, with a _item property
        """
        futures = list(futures)
        # Extract something pickleable out of the future items
        items = [(i, future._item) for i, future in enumerate(futures)]

        def _done_callback(indexed_item):
            """The batch callback function to mark stuff as done"""
            index, result = indexed_item
            futures[index].set_result(result)

        try:
            # Create a task instance that will wrap/unwrap our internal index
            task = _ParallelTask(func)

            batch_multi_node_parallel_map(
                func=task,
                iterable=items,
                nproc=self.config.nproc,
                njobs=self.config.njobs,
                cluster_method=self.config.method,
                chunksize=chunksize,
                callback=_done_callback,
            )
        except BaseException as e:
            for future in futures:
                # Technically could change state between done/set_exception
                # but don't need to worry about those instances
                if not future.done():
                    future.set_exception(e)

    def map(self, func, iterable, **kwargs):
        timeout = kwargs.pop("timeout", None)
        chunksize = kwargs.pop("chunksize", self.config.chunksize)

        assert timeout is None, "Cannot handle timeout"
        assert not kwargs, "Unexpected arguments: " + repr(kwargs)

        # Build future instances for each iterable
        futures = []
        for item in iterable:
            future = Future()
            future._item = item
            # We cannot cancel with easy_mp
            future.set_running_or_notify_cancel()
            futures.append(future)

        # If not specified, work out the chunking size here
        if chunksize is None:
            chunksize = _compute_chunksize(
                len(futures),
                self.config.njobs * self.config.nproc,
                self.config.min_chunksize,
            )

        thread = threading.Thread(
            target=self._threaded_do_parallel_map_call, args=(chunksize, func, futures)
        )
        thread.start()
        self._threads.append(thread)
        return futures

    def shutdown(self, wait=True):
        print("Shutdown wait", wait)
        if wait:
            for thread in self._threads:
                thread.join()
        else:
            for thread in self._threads:
                if thread.is_alive():
                    terminate_thread(thread)


if __name__ == "__main__":

    def func(x):
        return x

    iterable = list(range(100))

    multi_node_parallel_map(
        func,
        iterable,
        nproc=4,
        njobs=10,
        cluster_method="multiprocessing",
        callback=print,
    )
