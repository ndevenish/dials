from __future__ import absolute_import, division, print_function

import time
from concurrent.futures import wait

from dials.util.mp import BatchExecutor, multi_node_parallel_map


def _parallel_func(x):
    return x


def test_original_map_interface():
    """Copied original test from util/mp.py:__main__"""
    iterable = range(100)

    multi_node_parallel_map(
        _parallel_func,
        iterable,
        nproc=4,
        njobs=10,
        cluster_method="multiprocessing",
        callback=print,
    )


def _do_long_function(i):
    """Do a long function. Has to be pickleable so in root"""
    print("Starting", i)
    time.sleep(i)
    print(i)
    return i * 2


def test_batch_processor_easy_mp_passthrough():
    with BatchExecutor("easymp_multiprocessing", max_workers=5, njobs=1) as e:
        futs = []
        start = time.time()
        for i in range(5):
            futs.append(e.submit(_do_long_function, i))
        wait(futs)
        result = [f.result() for f in futs]
        assert result == [0, 2, 4, 6, 8]
        duration = time.time() - start
        assert duration < 7

        # Now test through the 'map' function
        start = time.time()
        result = e.map(_do_long_function, range(5))
        assert list(result) == [0, 2, 4, 6, 8]
        duration = time.time() - start
        assert duration < 7


# try:

# except KeyboardInterrupt:
#     pass

# breakpoint()
