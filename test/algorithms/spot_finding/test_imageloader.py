# coding: utf-8

from __future__ import absolute_import, division, print_function

from mock import Mock, call
import pytest
import threading

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


def test_max_loading(monkeypatch):
    entered_call = threading.Event()
    waiter = threading.Semaphore(0)

    def _semaphored_read(*args, **kwargs):
        """An image reading function that explicitly waits for permission"""
        # Once we've entered this call, we know where we should be
        print("read")
        # import pdb
        # pdb.set_trace()
        print("setting")
        entered_call.set()
        print("acquiring")
        waiter.acquire()
        return Mock()(*args, **kwargs)

    monkeypatch.setattr(imageloader, "_read_image_data", _semaphored_read)

    imageset = [[Mock(), Mock(), Mock()]]
    il = imageloader.AsyncImageLoader(imageset, maxsize=1)
    il.start()
    # import pdb
    # pdb.set_trace()
    # Wait for a thread to get to the read function before checking stuff
    while True:
        entered_call.wait(2)
        print("Main: Still waiting")
        import pdb

        pdb.set_trace()
    entered_call.clear()
    # We *Know* that a thread is either waiting or about to
    assert il.image_queue.empty()
