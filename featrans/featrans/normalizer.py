# coding: utf-8
import numpy
from collections import defaultdict

class MinMaxNorm(object):
    def __init__(self):
        self.initialized = defaultdict(bool)
        self.min_max = dict()

    def init(self, feaid, values):
        min_value = numpy.min(values)
        max_value = numpy.max(values)
        self.initialized[feaid] = True
        self.min_max[feaid] = (min_value, max_value)
    
    def __call__(self, feaid, value):
        """Normalize value by (value - min) / (max - min)
        """
        if not self.initialized[feaid]:
            raise ValueError("feature: {} not initialized.".format(feaid))
        min_value, max_value = self.min_max[feaid]
        return (value - min_value) / (max_value - min_value)


def create_normalizer(norm_type):
    if not norm_type:
        return None
    elif norm_type == "min_max":
        return MinMaxNorm()
    else:
        raise ValueError("Unsupported normalizer. Supported normalizers: min_max")