#! /usr/bin/env python
# -*- coding: utf-8 -*- 
"""
@time = 6/17/2016 5:56 PM
@author = Rongcheng
"""
import threading
import Queue
import time


class MultiThread(object):

    def __init__(self, function, args_list, max_thread=5, queue_results=False):
        self._function = function
        self._lock = threading.Lock()
        self._next_args = iter(args_list).next
        self._thread_pool = [threading.Thread(target=self._work) for i in range(max_thread)]

        if queue_results:
            self._queue = Queue.Queue()
        else:
            self._queue = None

    def _work(self):
        while True:
            self._lock.acquire()
            try:
                try:
                    args = self._next_args()
                except StopIteration:
                    break
            finally:
                self._lock.release()
            result = self._function(args)
            if self._queue is not None:
                self._queue.put((args, result))

    def get(self, *a, **kw):
        if self._queue is not None:
            return self._queue.get(*a, **kw)
        else:
            raise ValueError, 'Not queueing results'

    def start(self):
        for thread in self._thread_pool:
            time.sleep(1)
            thread.start()

    def join(self, timeout=None):
        for thread in self._thread_pool:
            thread.join(timeout)