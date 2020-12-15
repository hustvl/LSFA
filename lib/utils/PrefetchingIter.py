# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import mxnet as mx
from mxnet.io import DataDesc, DataBatch
import threading
import multiprocessing
import logging




def worker_loop(data_iter, key_queue, data_queue, shut_down, test_phase):
    key_queue.cancel_join_thread()
    data_queue.cancel_join_thread()
    while True:
        if shut_down.is_set():
            break
        if not test_phase:
            batch_str = key_queue.get()
            if batch_str is None:
                break
            data_iter.index = [int(batch_id) for batch_id in batch_str.split()]
            assert len(data_iter.index) == data_iter.batch_size
            data_iter.cur = 0
            data_queue.put(data_iter.next())
        else:
            batch_str = key_queue.get()
            if batch_str is None:
                break
            data_queue.put(data_iter.next())
    logging.info('goodbye')


class MultiThreadPrefetchingIter(mx.io.DataIter):
    def __init__(self, data_iter, num_workers=multiprocessing.cpu_count(), max_queue_size=8, test_phase=False):
        super(MultiThreadPrefetchingIter, self).__init__()
        if test_phase:
            num_workers = 1
        logging.info('num workers: %d' % num_workers)
        self.test_phase = test_phase
        self.data_iter = data_iter
        self.size = data_iter.size
        self.batch_size = data_iter.batch_size
        self.data_name = data_iter.data_name
        self.label_name = data_iter.label_name
        self.num_batches = self.size / self.batch_size
        assert self.size % self.batch_size == 0

        self.num_workers = num_workers
        self.workers = []
        self.cur = 0
        self.key_queue = mx.gluon.data.dataloader.Queue()
        self.data_queue = mx.gluon.data.dataloader.Queue(max_queue_size)
        self.key_queue.cancel_join_thread()
        self.data_queue.cancel_join_thread()
        self.shut_down = multiprocessing.Event()
        self._create_workers()

        import atexit
        atexit.register(lambda a: a.__del__(), self)

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    def _create_workers(self):
        for i in range(self.num_workers):
            worker = multiprocessing.Process(target=worker_loop,
                                             args=(self.data_iter, self.key_queue, self.data_queue, self.shut_down, self.test_phase))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _close_workers(self):
        for worker in self.workers:
            worker.join()
        self.workers = []

    def shutdown(self):
        self.shut_down.set()
        for i in range(len(self.workers)):
            self.key_queue.put(None)
        try:
            while not self.data_queue.empty():
                self.data_queue.get()
        except IOError:
            pass
        # self._close_workers()

    def __del__(self):
        self.shutdown()

    def reset(self):
        self.data_iter.reset()
        self.cur = 0

    def iter_next(self):
        return self.cur < self.num_batches

    def next(self):
        if self.cur == 0:
            index = self.data_iter.index.reshape((self.num_batches, self.data_iter.batch_size))
            for i in range(index.shape[0]):
                batch_str = '%d' % index[i, 0]
                for j in range(1, index.shape[1]):
                    batch_str += ' %d' % index[i, j]
                self.key_queue.put(batch_str)
        if self.iter_next():
            self.cur += 1
            return self.data_queue.get()
        else:
            raise StopIteration

class PrefetchingIter(mx.io.DataIter):
    """Base class for prefetching iterators. Takes one or more DataIters (
    or any class with "reset" and "next" methods) and combine them with
    prefetching. For example:

    Parameters
    ----------
    iters : DataIter or list of DataIter
        one or more DataIters (or any class with "reset" and "next" methods)
    rename_data : None or list of dict
        i-th element is a renaming map for i-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data
    rename_label : None or list of dict
        Similar to rename_data

    Examples
    --------
    iter = PrefetchingIter([NDArrayIter({'data': X1}), NDArrayIter({'data': X2})],
                           rename_data=[{'data': 'data1'}, {'data': 'data2'}])
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter ==1, "Our prefetching iter only support 1 DataIter"
        self.iters = iters
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = len(self.provide_data) * self.provide_data[0][0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for _ in range(self.n_iter)]
        self.next_batch = [None for _ in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])

    def reset(self):
        for e in self.data_ready:
            e.wait()
        for i in self.iters:
            i.reset()
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            return False
        else:
            self.current_batch = self.next_batch[0]
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad
