from sklearn.preprocessing import FunctionTransformer


class FunctionTransformerWithNames(FunctionTransformer):
    """
    Helper class that extends FunctionTransformer by adding 'get_feature_names'
    attribute. This attribute is a function that returns the column names of the last dataframe
    that was outputted by this transformer.

    Importantly, the function that is given to this transformer must have output with a
    'x.columns.values' attribute, such as a pandas dataframe. If the underlying function outputs a
    numpy array, this transformer will crash. In this case, it is reccommended to use the regular
    FunctionTransformer instead.

    Everything is unchanged in the FunctionTransformer, except that the default value of the
    'validate' parameter is changed to False. This mimics the behavior of sklearn from 0.22
    onwards. This is particularly important here, because validation turns pandas dataframes into
    numpy arrays, which means the column names are removed.

    Examples
    --------
    >>> from sam.utils import FunctionTransformerWithNames
    >>> from sam.feature_engineering import decompose_datetime
    >>> data = pd.DataFrame({'TIME': [1, 2, 3], 'VALUE': [4,5,6]})
    >>> transformer = FunctionTransformerWithNames(decompose_datetime,
    >>>                                            kw_args={'components': ['hour', 'minute']})
    >>> transformer.fit_transform(data)
    >>> transformer.get_feature_names()
    ['TIME', 'VALUE', 'TIME_hour', 'TIME_minute']
    """
    def __init__(self, func=None, inverse_func=None, validate=False,
                 accept_sparse=False, check_inverse=True,
                 kw_args=None, inv_kw_args=None):
        super(FunctionTransformerWithNames, self).\
            __init__(func=func, inverse_func=inverse_func, validate=validate,
                     accept_sparse=accept_sparse, check_inverse=check_inverse,
                     kw_args=kw_args, inv_kw_args=inv_kw_args)

    def transform(self, X, y=None):
        # y is completely ignored to be consistent with all versions of sklearn
        output = super(FunctionTransformerWithNames, self).transform(X)
        self._feature_names = list(output.columns.values)
        return output

    def get_feature_names(self):
        return self._feature_names
