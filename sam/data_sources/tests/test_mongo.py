import unittest

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pymongo.errors import ServerSelectionTimeoutError
from sam.data_sources import MongoWrapper


def find_mongo_host(possible_hostnames, port=27017):
    for hostname in possible_hostnames:
        mongo = MongoWrapper("test", "unittest", hostname, port, serverSelectionTimeoutMS=100)
        try:
            mongo.client.server_info()  # forces a call
        except ServerSelectionTimeoutError:  # no mongodb found
            continue  # try the next hostname
        return hostname  # Found a working hostname
    return None  # None of the hostnames worked


# These are the hostnames to try. Potentially add more in the future
possible_hostnames = ["localhost", "mongo", "mongodb"]
hostname = find_mongo_host(possible_hostnames, port=27017)

skipmongo = pytest.mark.skipif(
    hostname is None, reason="No valid mongodb on localhost/mongo/mongodb, port 27017"
)


class TestMongoWrapper(unittest.TestCase):
    def setUp(self):
        self.mongo = MongoWrapper("test", "unittest", hostname, 27017)

        self.dict_data = [{"some": 0, "thing": "bar"}, {"some": 1, "thing": "foo"}]
        self.dataframe = pd.DataFrame(
            {"some": [0, 1], "thing": ["bar", "foo"]}, columns=["some", "thing"]
        )
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
        expected = pd.concat([self.dataframe, self.dataframe], ignore_index=True)
        assert_frame_equal(result, expected)
        self.assertTrue(self.mongo.empty())

    @skipmongo
    def test_query(self):
        self.assertTrue(self.mongo.add(self.dict_data))
        result = self.mongo.get(query={"some": 1})
        expected = self.dataframe.iloc[1:]
        expected.index = [0]  # mongo doesn't remember index on query
        assert_frame_equal(result, expected)
        self.assertTrue(self.mongo.empty())

    @skipmongo
    def test_incorrect_data(self):
        # dict is not allowed, only list of dicts
        wrong = {"test": 3, "data": 4}
        self.assertRaises(TypeError, self.mongo.add, wrong)


if __name__ == "__main__":
    unittest.main()
