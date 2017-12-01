# coding: utf-8
from collections import Counter, defaultdict
import re
import numpy
import time
import sys

from utils import decorator


class LabelTransformer(object):
    @decorator.memory()
    def transform(self, text):
        text = text.strip()
        if not text:
            raise ValueError("Label must not be empty.")
        v = float(text)
        if v > 0:
            return 1
        else:
            return -1

class FeatureTransformer(object):
    def __init__(self):
        self.normalizer = None
        self.column_name = None
        self.fea2idx = dict()
        self.fea2values = defaultdict(list)
        self.num_features = 0
        self.num_samples = 0

    def _init_normalizer(self):
        if self.normalizer:
            for fea, values in self.fea2values.items():
                values = numpy.pad(values, (0, self.num_samples - len(values)), "constant", constant_values=0)
                feaid = self.fea2idx[fea]
                self.normalizer.init(feaid, values)
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self._init_normalizer()

    def load(self, column, init_feaidx):
        raise NotImplementedError

    def transform(self, text):
        raise NotImplementedError


class CategoryTransformer(FeatureTransformer):
    def __init__(self):
        super().__init__()

    def load(self, column):
        fea2idx = dict()
        for text in column:
            text = text.strip()
            if not text:
                continue
            if text not in fea2idx:
                idx = len(fea2idx)
                fea2idx[text] = idx

        self.column_name = column.name
        self.fea2idx = fea2idx
        self.num_features = len(fea2idx)
        self.num_samples = len(column)

    @decorator.memory()
    def transform(self, text):
        text = text.strip()
        if text in self.fea2idx:
            return [(self.fea2idx[text], 1)]
        else:
            return []


class TextTransformer(FeatureTransformer):
    def __init__(self, normalizer, sep):
        super().__init__()
        self.normalizer = normalizer
        self.sep_pattern = re.compile(sep) if sep else None

    def load(self, column):
        fea2idx = dict()
        fea2values = defaultdict(list)
        for text in column:
            terms = re.split(self.sep_pattern, text) if self.sep_pattern else [text]
            terms = (it.strip() for it in terms if it.strip())
            counter = Counter(terms)
            for term, count in counter.items():
                if term not in fea2idx:
                    fea2idx[term] = len(fea2idx)
                fea2values[term].append(count)

        assert len(fea2idx) == len(fea2values)
        self.fea2idx = fea2idx
        self.fea2values = fea2values
        self.column_name = column.name
        self.num_features = len(self.fea2idx)
        self.num_samples = len(column)
        self._init_normalizer()
    
    @decorator.memory()
    def transform(self, text):
        features = []
        terms = re.split(self.sep_pattern, text) if self.sep_pattern else [text]
        terms = (it.strip() for it in terms if it.strip())
        counter = Counter(terms)
        for term, count in counter.items():
            if term in self.fea2idx:
                feaid = self.fea2idx[term]
                value = count
                if self.normalizer:
                    value = self.normalizer(feaid, count)
                if value != 0:
                    features.append((feaid, value))
        return features

                
class NumericTransformer(FeatureTransformer):
    def __init__(self, normalizer, default_value):
        super().__init__()
        self.default_value = default_value
        self.normalizer = normalizer

    def load(self, column):
        values = []
        for text in column:
            text = text.strip()
            value = self.default_value
            if text:
                value = float(text)
            values.append(value)

        self.fea2idx[column.name] = 0
        self.fea2values[column.name] = values
        self.column_name = column.name
        self.num_features = len(self.fea2idx)
        self.num_samples = len(column)
        self._init_normalizer()
    
    @decorator.memory()
    def transform(self, text):
        text = text.strip()
        value = self.default_value
        feaid = self.fea2idx[self.column_name]
        if text:
            value = float(text)
        if self.normalizer:
            value = self.normalizer(feaid, value)
        if value != 0:
            return [(feaid, value)]
        else:
            return []
        