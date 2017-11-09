# coding: utf-8
import unittest
import pandas
import os

from featrans.feaengine import FeaEngine
from featrans.normalizer import MinMaxNorm

class TestFeaengine(unittest.TestCase):
    def setUp(self):
        self.data = {
            "label": pandas.Series(["1", "1", "0", "1", "1"]),
            "fea1": pandas.Series(["a,a,c", "a,c,d", "c,e,f", "a,f", ""]),
            "fea2": pandas.Series(["1", "2", "3", "", "3"])
        }
        self.df = pandas.DataFrame(self.data, columns=["label", "fea1", "fea2", "ignore"])
        self.transformers = [
            ("label", "label"), 
            ("fea1", "text", "min_max", ","),
            ("fea2", "num", "min_max", 1.0)
        ]
        self.raw_feas = [
            (1, [(0,2), (1,1), (5,1.0)]),
            (1, [(0,1), (1,1), (2,1), (5,2)]),
            (-1, [(1,1), (3,1), (4,1), (5,3)]),
            (1, [(0,1), (4,1), (5,1)]),
            (1, [(5,3)])
        ]
        self.min_max_feas = [
            (1, [(0,1), (1,1)]),
            (1, [(0,0.5), (1,1), (2,1), (5,0.5)]),
            (-1, [(1,1), (3,1), (4,1), (5,1)]),
            (1, [(0,0.5), (4,1)]),
            (1, [(5,1)])
        ]
        self.feaengine = FeaEngine(None)

    def test_set_transformers(self):
        self.feaengine.set_transformers(self.transformers)
        self.assertEqual(3, len(self.feaengine.transformers))
    
    def test_transform(self):
        self.feaengine.set_transformers(self.transformers)
        self.feaengine.load(self.df)
        self.__internal_test_transform(self.feaengine, True)
        self.__internal_test_transform(self.feaengine, False)
    
    def __internal_test_transform(self, engine, is_minmax):
        for feaname, _ in self.data.items():
            if feaname not in ["label", "ignore"]:
                normalizer = MinMaxNorm() if is_minmax else None
                engine.set_normalizer(feaname, normalizer)
        feas = self.min_max_feas if is_minmax else self.raw_feas
        for (label, features), (expected_label, expected_features) in zip(engine.transform(self.df), feas):
            self.assertEqual(label, expected_label)
            self.assertListEqual(features, expected_features)

    def test_save_load(self):
        f = "test.txt"
        self.feaengine.set_transformers(self.transformers)
        self.feaengine.load(self.df)
        self.feaengine.save_engine(f)
        feaengine2 = FeaEngine(None)
        feaengine2.load_engine(f)
        self.__internal_test_transform(feaengine2, True)
        self.__internal_test_transform(feaengine2, False)
        os.remove(f)