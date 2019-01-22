import unittest
from sam.utils import MongoWrapper
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pymongo.errors import ServerSelectionTimeoutError


def skip_pymongo_tests():
    mongo = MongoWrapper('test', 'unittest', 'localhost', 27017,
                         serverSelectionTimeoutMS=100)
    try:
        mongo.client.server_info()  # forces a call
    except ServerSelectionTimeoutError:  # no mongodb found
        return True
    return False

skipmongo = pytest.mark.skipif(skip_pymongo_tests(), reason="No valid mongodb on localhost")


class TestMongoWrapper(unittest.TestCase):

    def setUp(self):
        self.mongo = MongoWrapper('test', 'unittest', 'localhost', 27017)

        self.dict_data = [{'some': 0, 'thing': 'bar'},
                          {'some': 1, 'thing': 'foo'}]
        self.dataframe = pd.DataFrame({'some': [0, 1],
                                       'thing': ['bar', 'foo']},
                                      columns=['some', 'thing'])
        # empty the collection before testing
        self.mongo.empty()

    @skipmongo
    def test_dict_data(self):
        self.assertTrue(self.mongo.add(self.dict_data))
        result = self.mongo.get()
        assert_frame_equal(result, self.dataframe)

        # query a dictionary
        result = self.mongo.get(as_df=False)
        self.assertEqual(result, self.dict_data)

        self.assertTrue(self.mongo.empty())

    @skipmongo
    def test_add_data(self):
        self.assertTrue(self.mongo.add(self.dict_data))
        self.assertTrue(self.mongo.add(self.dataframe))
        result = self.mongo.get()
        expected = pd.concat([self.dataframe, self.dataframe],
                             ignore_index=True)
        assert_frame_equal(result, expected)
        self.assertTrue(self.mongo.empty())

    @skipmongo
    def test_query(self):
        self.assertTrue(self.mongo.add(self.dict_data))
        result = self.mongo.get(query={'some': 1})
        expected = self.dataframe.iloc[1:]
        expected.index = [0]  # mongo doesn't remember index on query
        assert_frame_equal(result, expected)
        self.assertTrue(self.mongo.empty())

    @skipmongo
    def test_incorrect_data(self):
        # dict is not allowed, only list of dicts
        wrong = {"test": 3, "data": 4}
        self.assertRaises(TypeError, self.mongo.add, wrong)
