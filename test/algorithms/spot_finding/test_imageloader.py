# coding: utf-8

from __future__ import absolute_import, division, print_function

from mock import Mock, call
import pytest
import threading
import time

import dials.algorithms.spot_finding.imageloader as imageloader


@pytest.fixture
def mock_read_data(monkeypatch):
    """Prevents reading real data with the image loader"""
    func = Mock()
    monkeypatch.setattr(imageloader, "_read_image_data", func)
    return func


@pytest.fixture
def single_thread(monkeypatch):
    """Runs any threading launch sequentially"""

    def _fake_thread(_, f, *args, **kwargs):
        f(*args, **kwargs)
        return Mock()(f, *args, **kwargs)

    monkeypatch.setattr(threading, "Thread", _fake_thread)


def test_empty_setup():
    il = imageloader.AsyncImageLoader([])
    with pytest.raises(AssertionError):
        il.start()


def test_initial_queue_creation():
    test_imageset = [[Mock(), Mock(), Mock()]]
    il = imageloader.AsyncImageLoader(test_imageset)
    assert il._image_indices.qsize() == 3


def test_load_called(single_thread, mock_read_data):
    test_imageset = [[Mock(), Mock(), Mock()]]
    il = imageloader.AsyncImageLoader(test_imageset)
    il.start()
    mock_read_data.assert_has_calls(
        [
            call(test_imageset[0], 1),
            call(test_imageset[0], 0),
            call(test_imageset[0], 2),
        ],
        any_order=True,
    )


def test_async_read(monkeypatch):
    entered_call = threading.Event()
    waiter = threading.Semaphore(0)

    def _semaphored_read(*args, **kwargs):
        """An image reading function that explicitly waits for permission"""
        # Once we've entered this call, we know where we should be
        print("setting")
        entered_call.set()
        print("acquiring")
        waiter.acquire()
        print("Returning read", args)
        return Mock()(*args, **kwargs)

    monkeypatch.setattr(imageloader, "_read_image_data", _semaphored_read)

    imageset = [[Mock(), Mock(), Mock()]]
    il = imageloader.AsyncImageLoader(imageset)  # , maxsize=1
    il.start()
    # Wait for a thread to get to the read function before checking stuff
    # Wait, but since a test assert on wait time.
    assert entered_call.wait(2)
    entered_call.clear()
    # We *Know* that a thread is either waiting or about to
    assert il.image_queue.empty()
    # Let it continue, then assert that one item has been added and removed
    waiter.release()
    assert entered_call.wait(2)
    assert il.image_queue.qsize() == 1
    # assert il.image_queue.qsize() == 1
    # Let the end of the loop run
    waiter.release()
    waiter.release()
    il.join()


def test_max_loading(monkeypatch):
    # entered_call = threading.Event()
    waiter = threading.Semaphore(0)

    def _notifying_read(*args, **kwargs):
        """An image reading function that explicitly releases permission"""
        print("releasing")
        waiter.release()
        print("Returning read", args)
        return Mock()(*args, **kwargs)

    monkeypatch.setattr(imageloader, "_read_image_data", _notifying_read)

    imageset = [[Mock(), Mock(), Mock()]]
    il = imageloader.AsyncImageLoader(imageset, maxsize=1)
    il.start()
    # First time round we have done nothing
    waiter.acquire()
    # Then, we get released again on the second read function
    waiter.acquire()

    assert il.image_queue.qsize() == 1
    time.sleep(0.2)
    assert il.image_queue.qsize() == 1
    # Pull from the queue and wait until we go round again
    assert il.image_queue.qsize() == 1
    assert il.image_queue.get()
    waiter.acquire()
    assert il.image_queue.qsize() == 1
    assert il.image_queue.get()
    assert il.image_queue.get()
    # Mark tasks as finished
    il.image_queue.task_done()
    il.image_queue.task_done()
    il.image_queue.task_done()
    # Wait for the runner to rejoin
    il.join()


def test_late_adding(mock_read_data):
    one, two, three, four = Mock(), Mock(), Mock(), Mock()
    imageset = [one, two, three, four]
    il = imageloader.AsyncImageLoader()
    il.add(imageset)
    il.start()
    il.join()
    assert il.image_queue.qsize() == 4

    il = imageloader.AsyncImageLoader()
    il.add(imageset, 1)
    il.start()
    il.join()
    assert il.image_queue.qsize() == 1

    il = imageloader.AsyncImageLoader()
    il.add(imageset, (None, 2))
    il.start()
    il.join()
    assert il.image_queue.qsize() == 2

    il = imageloader.AsyncImageLoader()
    il.add(imageset, (1, None))
    il.start()
    il.join()
    assert il.image_queue.qsize() == 3
