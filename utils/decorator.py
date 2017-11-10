# coding: utf-8
from functools import wraps
from collections import OrderedDict
import time


def memory(fn):
    """Used to cache function result.
    Example:
    @memory
    def compute(*args):
        return time_cost_compute(*args)
    """
    @wraps(fn)
    def wrapper(*args):
        if args not in wrapper.cache:
            wrapper.cache[args] = fn(*args)
        return wrapper.cache[args]
    wrapper.cache = dict()
    return wrapper


def collector(c_func):
    """
    Example:
    @collector(set)
    def compute(*args):
        for i in range(5):
            yield i
    compute will return set(0, 1, 2, 3, 4)
    """
    def main(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return c_func(fn(*args, **kwargs))
        return wrapper
    return main


def counter(key_func):
    """Count the args that call function.
    @counter(lambda x: x[0])
    def test(x):
        return x
    test(1)
    test(1)
    test(2)
    test.counter([1]) -> 2
    """
    def main(fn):
        @wraps(fn)
        def wrapper(*args):
            key = key_func(args)
            #wrapper.counter[key] += 1
            wrapper.counter[key] = wrapper.counter.get(key, 0) + 1
            return fn(*args)
        #wrapper.counter = defaultdict(lambda : 0)
        wrapper.counter = OrderedDict()
        return wrapper
    return main


def timer(fn):
    """Statistic the function runtime in seconds.
    """
    def wrapper(*args):
        start = time.time()
        result = fn(*args)
        end = time.time()
        wrapper.cost += (end - start)
        return result
    wrapper.cost = 0
    return wrapper

def yield_timer(fn):
    """Statistic the function runtime in seconds.
    For function with yield.
    """
    def wrapper(*args):
        start = time.time()
        for i in fn(*args):
            end = time.time()
            wrapper.cost += end - start
            yield i
            start = time.time()
        end = time.time()
        wrapper.cost += (end - start)
    wrapper.cost = 0
    return wrapper