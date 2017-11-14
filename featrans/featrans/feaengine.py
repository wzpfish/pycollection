#coding: utf-8
import logging
import time
import _pickle as pickle

from .transformer import *
from .normalizer import *

class FeaEngine(object):
    def __init__(self):
        self.transformers = None
        self.index_from = 0
        self.columns = None

        self.idx2feaname = dict()
        self.feaname_updated = False
    
    def update_engine(self, transformers, index_from, columns):
        """Update the whole engine.
        Params:
        transformers: list of transformer settings. e.g. [("Label", "label")]
        index_from: index feature from(0 or 1 for common).
        columns: columns which need to transform, including label column. 
        """
        self.update_transformers(transformers).update_index_from(index_from).update_columns(columns)
    
    def update_transformers(self, transformers):
        if transformers:
            self.__set_transformers(transformers)
        return self
        
    def update_index_from(self, index_from):
        if index_from is not None:
            self.index_from = index_from
        return self

    def update_columns(self, columns):
        if columns:
            self.columns = columns
        return self
        
    def __update_feaname(self):
        idx2feaname = dict()
        index_from = self.index_from
        for column_name in self.columns:
            transformer = self.transformers[column_name]
            if isinstance(transformer, LabelTransformer):
                continue
            for feaname, idx in transformer.fea2idx.items():
                idx2feaname[idx + index_from] = "{}-{}".format(column_name, feaname)
            index_from += transformer.num_features
        self.idx2feaname = idx2feaname
        self.feaname_updated = True

    def save_engine(self, save_to):
        with open(save_to, "wb") as fout:
            pickle.dump(self.transformers, fout)

    def load_engine(self, filename):
        with open(filename, "rb") as fin:
            self.transformers = pickle.load(fin)

    def __set_transformers(self, params):
        """Create transformers for each column defined in params.
        """
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
        """Load data of pandas format to prepare transformers.
        """
        fea_count = 0
        for feaname in df:
            transformer = self.transformers.get(feaname, None)
            if not transformer or isinstance(transformer, LabelTransformer):
                continue
            transformer.load(df[feaname])
            fea_count += transformer.num_features
            logging.info("Load %d features for column %s", transformer.num_features, feaname)
        logging.info("%d features loaded.", fea_count)

    def transform_row(self, row):
        label = None
        features = []

        index_from = self.index_from
        for column_name in self.columns:
            transformer = self.transformers[column_name]
            text = row[column_name]
            if isinstance(transformer, LabelTransformer):
                label = transformer.transform(text)
            else:
                reindexed = []
                for idx, value in transformer.transform(text):
                    reindexed.append((idx + index_from, value))
                features.extend(reindexed)
                index_from += transformer.num_features
        return label, features
    
    def __check_valid(self):
        if not self.columns:
            raise ValueError("Must call 'update_columns' first to set needed label and feature columns.")
        if not self.transformers:
            raise ValueError("Must call 'update_transformers' first to set tranformers.")
        if not self.index_from:
            raise ValueError("Must call 'update_index' first to set init index.")

    def transform(self, df):
        """Transform features for given columns.
        """
        self.__check_valid()
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Duplicate columns not allowed.")
        for column_name in self.columns:
            if column_name not in self.transformers:
                raise ValueError("No transformer found for column: {}.".format(column_name))
            if column_name not in df.columns.values:
                raise ValueError("No column named {} found for df.".format(column_name))
        
        for _, row in df.iterrows():
            yield self.transform_row(row)
        
    def feature_name(self, fea_idx):
        """Return feature name of given feature index and need columns.
        Return None if no such feature index found.
        """
        self.__check_valid()
        if not self.feaname_updated:
            self.__update_feaname()

        return self.idx2feaname.get(fea_idx, None)

    def summary(self):
        self.__check_valid()
        
        stats = []
        index_from = self.index_from
        for column_name in self.columns:
            transformer = self.transformers[column_name]
            if isinstance(transformer, LabelTransformer):
                continue
            stats.append("column: {}, id: [{}, {}]".format(column_name, index_from, index_from + transformer.num_features - 1))
            index_from += transformer.num_features
        return "\n".join(stats)