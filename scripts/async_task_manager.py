import threading
import time


class AsyncTaskManager:
    def __init__(self, target, args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.condition = threading.Condition()
        self.result = None
        self.thread = threading.Thread(target=self.worker)
        self.stopped = False
        self.thread.daemon = True
        self.thread.start()

    def worker(self):
        while True:
            self.condition.acquire()
            while self.result is not None:
                if self.stopped:
                    self.condition.release()
                    return
                self.condition.notify()
                self.condition.wait()
            self.condition.notify()
            self.condition.release()

            result = (self.target(*self.args, **self.kwargs), )

            self.condition.acquire()
            self.result = result
            self.condition.notify()
            self.condition.release()

    def get_next(self):
        self.condition.acquire()
        while self.result is None:
            self.condition.notify()
            self.condition.wait()
        result = self.result[0]
        self.result = None
        self.condition.notify()
        self.condition.release()
        return result

    def stop(self):
        while self.thread.is_alive():
            self.condition.acquire()
            self.stopped = True
            self.condition.notify()
            self.condition.release()


def task():
    print('begin sleeping...')
    time.sleep(1)
    print('end sleeping.')
    task.i += 1
    print('returns', task.i)
    return task.i


task.i = 0

if __name__ == '__main__':
    async = AsyncTaskManager(task)
    t = time.time()
    for i in range(5):
        ret = async.get_next()
        # ret = task()
        print('got', ret)
        time.sleep(1)
    async.stop()
    print(time.time() - t)
