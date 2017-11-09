#coding: utf-8
import logging
import time
import _pickle as pickle

from .transformer import *
from .normalizer import *

class FeaEngine(object):
    def __init__(self, transformers):
        self.transformers = dict()
        if transformers:
            self.set_transformers(transformers)
    
    def save_engine(self, save_to):
        with open(save_to, "wb") as fout:
            pickle.dump(self.transformers, fout)

    def load_engine(self, filename):
        with open(filename, "rb") as fin:
            self.transformers = pickle.load(fin)

    def set_transformers(self, params):
        transformers = dict()
        for param in params:
            column_name = param[0]
            if param[1] == "label":
                transformers[column_name] = LabelTransformer()
            elif param[1] == "num":
                norm_type = param[2]
                default = float(param[3])
                norm = create_normalizer(norm_type)
                transformers[column_name] = NumericTransformer(norm, default)
            elif param[1] == "text":
                norm_type = param[2]
                sep = param[3]
                norm = create_normalizer(norm_type)
                transformers[column_name] = TextTransformer(norm, sep)
            elif param[1] == "category":
                transformers[column_name] = CategoryTransformer()
            else:
                raise ValueError("Unsupported transformer type {}. Supported types are: label, num, text, category.".format(param[1]))
        self.transformers = transformers

    def set_normalizer(self, column_name, normalizer):
        if column_name not in self.transformers:
            raise ValueError("No transformer found for column: {}".format(column_name))
        if type(self.transformers[column_name]) is LabelTransformer:
            raise ValueError("LabelTransformer do not need normalizer.")
        self.transformers[column_name].set_normalizer(normalizer)

    def load(self, df):
        init_feaidx = 0
        for feaname in df:
            transformer = self.transformers.get(feaname, None)
            if not transformer or type(transformer) is LabelTransformer:
                continue
            transformer.load(df[feaname], init_feaidx)
            init_feaidx += transformer.num_features
            logging.info("Load %d features for column %s", transformer.num_features, feaname)
        logging.info("%d features loaded.", init_feaidx)

    def transform_row(self, row):
        label = None
        features = []
        for feaname, text in row.items():
            transformer = self.transformers.get(feaname, None)
            if not transformer:
                continue
            if type(transformer) is LabelTransformer:
                label = transformer.transform(text)
            else:
                features.extend(transformer.transform(text))
        return label, features

    def transform(self, df):
        for _, row in df.iterrows():
            yield self.transform_row(row)       