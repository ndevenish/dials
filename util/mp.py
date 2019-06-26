from __future__ import absolute_import, division, print_function

import ctypes
import logging
import math
import multiprocessing
import os
import pickle
import threading
import time
import warnings
from collections import namedtuple
from concurrent.futures import Executor, Future, ThreadPoolExecutor

import future.moves.itertools as itertools
import psutil
import six.moves.queue as queue

import libtbx.easy_mp

from dials.util.cluster_map import cluster_map as drmaa_parallel_map

logger = logging.getLogger(__name__)


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


def BatchExecutor(method="multiprocessing", max_workers=None, njobs=1, **kwargs):
    """
    Execute tasks in parallel via a variety of methods.

    Args:
        method:
            The method to use. If None, then "multiprocessing" will be
            used if nprocs is None or nprocs > 1, otherwise "threads"
            will be used with a single executor to effectively process
            the jobs serially.
        nprocs:
            The number of tasks to process concurrently. If nprocs is
            None or not given, it will default depending on the method:
                threads:
                    The number of processors on the machine, multiplied
                    by 5, assuming that ThreadPoolExecutor is often used
                    to overlap I/O instead of CPU work and the number of
                    workers should be higher than the number of workers
                    for multiprocessing.
                multiprocessing:
                    It will default to the number of processors on the machine.
                other:
                    It will default to 1, assuming that njobs are being specified.
    """
    # Work out default methods
    if not method and max_workers == 1:
        method = "threads"
    elif not method:
        method = "multiprocessing"
    # Work out default number of workers
    if method == "threads" and not max_workers:
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    elif method == "multiprocessing" and not max_workers:
        max_workers = multiprocessing.cpu_count()
    elif not max_workers:
        max_workers = 1

    # Validate njobs
    if method in {"threads", "multiprocessing"} and njobs != 1:
        raise ValueError("Can not specify njobs with method '{}'".format(method))

    if method == "threads":
        print("DEBUG: Running tasks in threads n={}".format(max_workers))
        return ThreadPoolExecutor(max_workers=max_workers)
    elif method == "serial":
        assert max_workers == 1 and njobs == 1
        print("DEBUG: Running tasks serially on submit")
        return _SerialExecutor()

    print("DEBUG: Running tasks with parallel map wrapper")
    return _WrapBatchParallelMap(method, max_workers, njobs, **kwargs)


class _SerialExecutor(Executor):
    """Shim executor that runs everything serially"""

    def submit(self, fn, *args, **kwargs):
        # For debugging, try to pickle function
        pickle.dumps(fn)

        fut = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)
        return fut


def _do_dispatch_parallel_task(bundled_args):
    """Pickleable front-end interface to dispatching batch tasks"""
    i, (func, args, kwargs) = bundled_args
    return (i, func(*args, **kwargs))


class _WrapBatchParallelMap(Executor):

    """Wrapper to abstract away dealing with easy_mp and multiprocessing."""

    def __init__(self, method, nprocs, njobs=1, chunksize=None, min_chunksize=20):
        if chunksize is libtbx.Auto:
            chunksize = None
        self.config = MPConfig(method, njobs, nprocs, chunksize, min_chunksize)
        self._thread = None
        self._tasks = queue.Queue()
        self._shutdown = False

    def _threaded_do_check_task_queue(self):
        """Called in a thread. Calls the parallel map function.

        Args:
            chunksize: Individual submissions can override the configuration
            func: The function to call
            futures: The Future objects, with a _item property
        """
        while not self._shutdown:
            print("DEBUG: Checking for tasks")
            # Once started, wait a second to give time to fill the queue
            time.sleep(1)
            # Get the associated lists of task, future
            tasks = []
            futures = []
            try:
                while True:
                    future, task = self._tasks.get_nowait()
                    if future.set_running_or_notify_cancel():
                        futures.append(future)
                        tasks.append((len(tasks), task))
            except queue.Empty:
                pass
            if not tasks:
                continue
            print("DEBUG: Picking up {} tasks".format(len(tasks)))

            futures = list(futures)

            def _done_callback(indexed_item):
                """The batch callback function to mark stuff as done"""
                index, result = indexed_item
                futures[index].set_result(result)

            try:
                chunksize = self.config.chunksize
                if not chunksize:
                    chunksize = _compute_chunksize(
                        len(tasks), self.config.nproc, self.config.min_chunksize
                    )
                batch_multi_node_parallel_map(
                    func=_do_dispatch_parallel_task,
                    iterable=tasks,
                    nproc=self.config.nproc,
                    njobs=self.config.njobs,
                    cluster_method=self.config.method,
                    chunksize=chunksize,
                    callback=_done_callback,
                )
            except BaseException as exc:
                for future in futures:
                    # Technically could change state between done/set_exception
                    # but don't need to worry about those instances
                    if not future.done():
                        future.set_exception(exc)
        print("DEBUG: Ending task thread loop")

    def _start_thread_processing(self):
        if not self._thread:
            self._thread = threading.Thread(target=self._threaded_do_check_task_queue)
            self._thread.start()

    def submit(self, fn, *args, **kwargs):
        future = Future()
        self._tasks.put((future, (fn, args, kwargs)))
        self._start_thread_processing()
        return future

    def shutdown(self, wait=True):
        # Cancel all pending tasks
        try:
            while True:
                task, _ = self._tasks.get_nowait()
                task.cancel()
        except queue.Empty:
            pass
        # Decide how to shut down the processing
        if wait:
            self._shutdown = True
            if self._thread:
                self._thread.join()
        else:
            if self._thread and self._thread.is_alive():
                terminate_thread(self._thread)
