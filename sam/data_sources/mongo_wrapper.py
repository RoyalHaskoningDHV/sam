import pandas as pd


class MongoWrapper:
    """
    Provides a simple wrapper to the MongoDB layers

    This class provides a wrapper for basic functionality in MongoDB. We aim
    to use MongoDB as storage layer between analyses and e.g. dashboarding.

    Parameters
    ----------
    db: string
        Name of the database
    collection: string
        the name of the collection to fetch
    location: string, optional (default="localhost")
        Location of the database
    port: integer, optional (default=27017)
        Port that the database is reachable on
    **kwargs: arbitrary keyword arguments
        Passed through to `pymongo.MongoClient`

    Examples
    --------
    >>> from sam.data_sources import MongoWrapper  # doctest: +SKIP
    >>> mon = MongoWrapper('test_magweg','test_magookweg')  # doctest: +SKIP
    >>> mon.empty().add([{'test': 7}]).get()  # doctest: +SKIP
    >>> test  # doctest: +SKIP
    """

    def __init__(self, db, collection, location="localhost", port=27017, **kwargs):
        import pymongo  # Only needed now

        self.client = pymongo.MongoClient(location, port, **kwargs)
        self.db = self.client[db]
        self.collection = self.db[collection]

    def get(self, query={}, as_df=True):
        """Get as specific collection from the database

        Parameters
        ----------
        query: dictionary-like, optional (default={})
            dictionary of parameters to use in the query.
            e.g. { "address": "Park Lane 38" }
        as_df: boolean, optional (default=True)
            return the query results as a Pandas Dataframe

        Returns
        -------
        result : pandas dataframe, or list of dictionaries
            the results of the query
        """
        col = list(self.collection.find(query))

        if as_df:
            col = pd.DataFrame(col)
            col = col.drop("_id", axis=1)

        return col

    def add(self, content):
        """Get as specific collection from the database

        Parameters
        ----------
        content: list of dictionaries, or pandas dataframe
            list of items to add to the collection

        Returns
        -------
        result : self
        """
        if isinstance(content, pd.DataFrame):
            content = content.to_dict(orient="records")

        if self.collection.insert_many(content):
            return self
        else:
            return False

    def empty(self):
        """Empty the collection

        Returns
        -------
        result: self
        """
        if self.collection.delete_many({}):
            return self
        else:
            return False
