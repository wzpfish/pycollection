# coding: utf-8
from functools import wraps
import os
import math
import logging
import traceback
import logging

logger = logging.getLogger(__name__)

#------------------multithreading------------------------------
def argmerge(fn):
    @wraps(fn)
    def wrapper(args):
        return fn(*args)
    return wrapper


def merge_file(concurrency, split_path_pattern, fout):
    for i in range(concurrency):
        path = split_path_pattern % i
        with open(path, "r", encoding='utf-8') as fin:
            for line in fin:
                fout.write(line[:-1] + "\n")
        os.remove(path)

@argmerge
def worker_file(pid, fn, args, concurrency, split_path_pattern):
    with open(split_path_pattern % pid, 'w', encoding='utf-8') as fout:
        num = int(math.ceil(len(args) / float(concurrency)))
        for i in range(num*pid, min(num*(pid+1), len(args))):
            if (i-num*pid+1) % 100 == 0:
                logger.info('pid=%d num=%d/id=%d' % (pid, num, i-num*pid+1))
            fn(fout, *args[i])

def scheduler_file(fn, args, concurrency, split_path_pattern, fout):
    from multiprocessing.dummy import Pool
    pool = Pool(concurrency)
    try:
        pool.map(worker_file, [(i, fn, args, concurrency, split_path_pattern) for i in range(concurrency)])
    except:
        logger.error(traceback.format_exc())
    merge_file(concurrency, split_path_pattern, fout)