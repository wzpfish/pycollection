# coding: utf-8
import unittest
import pandas 

from featrans.transformer import TextTransformer, NumericTransformer, CategoryTransformer
from featrans.normalizer import MinMaxNorm


class TestTextTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = TextTransformer(None, ",| ")
        self.column = pandas.Series(["a,a,c", "a,c,d", "c,e,f", "a,f", ""])

    def test_load(self):
        self.transformer.load(self.column, 0)
        self.assertEqual(5, self.transformer.num_features)
        self.assertEqual(5, self.transformer.num_samples)
    
    def test_transform(self):
        cases = [
            ("a,a,c", [(0,2),(1,1)], [(0,1.0),(1,1.0)]),
            ("a,c,d", [(0,1),(1,1),(2,1)], [(0,0.5),(1,1.0),(2,1.0)]),
            ("c,e,f", [(1,1),(3,1),(4,1)], [(1,1.0),(3,1.0),(4,1.0)]),
            ("a,f", [(0,1),(4,1)], [(0,0.5),(4,1.0)]),
            ("", [], [])
        ]

        self.transformer.load(self.column, 0)
        self.transformer.set_normalizer(None)
        for text, expected, _ in cases:
            features = self.transformer.transform(text)
            self.assertListEqual(expected, features)
        
        self.transformer.set_normalizer(MinMaxNorm())
        for text, _, expected in cases:
            features = self.transformer.transform(text)
            self.assertListEqual(expected, features)


class TestNumericTransformer(unittest.TestCase):
    def setUp(self):
        self.d = 1
        self.transformer = NumericTransformer(None, self.d)
        self.column = pandas.Series(["1", "2", "3", "", "3"])
    
    def test_load(self):
        self.transformer.load(self.column, 5)
        self.assertEqual(1, self.transformer.num_features)
        self.assertEqual(5, self.transformer.fea2idx[self.column.name])

    def test_transform(self):
        cases = [
            ("1", [(0,1)], []),
            ("2", [(0,2)], [(0,0.5)]),
            ("3", [(0,3)], [(0,1.0)]),
            ("", [(0,self.d)], []),
            ("3", [(0,3)], [(0,1.0)])
        ]
        self.transformer.load(self.column, 0)
        self.transformer.set_normalizer(None)
        for text, expected, _ in cases:
            features = self.transformer.transform(text)
            self.assertListEqual(expected, features)
        
        self.transformer.set_normalizer(MinMaxNorm())
        for text, _, expected in cases:
            features = self.transformer.transform(text)
            self.assertListEqual(expected, features)

class TestCategoryTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = CategoryTransformer()
        self.column = pandas.Series(["1", "2", "", "3"])

    def test_load(self):
        self.transformer.load(self.column, 0)
        self.assertEqual(3, self.transformer.num_features)
    
    def test_transform(self):
        init_idx = 3
        cases = [
            ("1", [(init_idx+0,1)]),
            ("2", [(init_idx+1,1)]),
            ("3", [(init_idx+2,1)]),
            ("", [])
        ]
        self.transformer.load(self.column, init_idx)
        for text, expected in cases:
            features = self.transformer.transform(text)
            self.assertListEqual(expected, features)
        
if __name__ == "__main__":
    unittest.main()