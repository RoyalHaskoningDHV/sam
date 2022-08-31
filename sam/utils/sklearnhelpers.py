from typing import Callable, List

from sklearn.preprocessing import FunctionTransformer


class FunctionTransformerWithNames(FunctionTransformer):
    """
    Helper class that extends FunctionTransformer by adding 'get_feature_names'
    method. This method returns the column names of the last dataframe
    that was returned by this transformer.

    Importantly, the function that is given to this transformer must have an output x with
    attributes 'x.columns.values', such as a pandas dataframe. If the underlying function outputs a
    numpy array, this transformer will crash. In this case, it is reccommended to use the regular
    FunctionTransformer instead.

    Everything is unchanged in the FunctionTransformer, except that the default value of the
    'validate' parameter is changed to False. This mimics the behavior of sklearn from 0.22
    onwards. This is particularly important here, because validation turns pandas dataframes into
    numpy arrays, which means that the column names are removed.

    Parameters
    ----------
    func : callable, optional default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.
    inverse_func : callable, optional default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.
    validate : bool, optional default=False
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:
        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.
    accept_sparse : bool, optional
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.
    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.
    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.
    inv_kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to inverse_func.

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.utils import FunctionTransformerWithNames
    >>> from sam.feature_engineering import decompose_datetime
    >>> data = pd.DataFrame({'TIME': [1, 2, 3], 'VALUE': [4,5,6]})
    >>> data['TIME'] = pd.to_datetime(data['TIME'])
    >>> transformer = FunctionTransformerWithNames(decompose_datetime,
    ...                                            kw_args={'components': ['hour', 'minute']})
    >>> transformer.fit_transform(data)
                               TIME  VALUE  TIME_hour  TIME_minute
    0 1970-01-01 00:00:00.000000001      4          0            0
    1 1970-01-01 00:00:00.000000002      5          0            0
    2 1970-01-01 00:00:00.000000003      6          0            0
    >>> transformer.get_feature_names_out()
    ['TIME', 'VALUE', 'TIME_hour', 'TIME_minute']
    """

    def __init__(
        self,
        func: Callable = None,
        inverse_func: Callable = None,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = True,
        kw_args: dict = None,
        inv_kw_args: dict = None,
    ):
        super(FunctionTransformerWithNames, self).__init__(
            func=func,
            inverse_func=inverse_func,
            validate=validate,
            accept_sparse=accept_sparse,
            check_inverse=check_inverse,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )

    def transform(self, X, y=None):
        """
        Applies the function, and saves the output feature names
        """
        # y is completely ignored to be consistent with all versions of sklearn
        output = super(FunctionTransformerWithNames, self).transform(X)
        self._feature_names = list(output.columns.values)
        return output

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns the feature names saved during `transform`
        """
        return self._feature_names
