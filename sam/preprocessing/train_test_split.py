from typing import Union, Optional
import pandas as pd


def datetime_train_test_split(
    *arrays: Union[pd.DataFrame, pd.Series],
    datetime: str,
    datecol: Optional[str] = None,
):
    """
    Split the dataframe into train and test sets based on datetime index values

    Parameters
    ----------
    arrays : Union[pd.DataFrame, pd.Series]
        Allowed inputs are pandas dataframes or series (with datetime index)
    datetime : str
        Datetime to split the dataframe on
    datecol : str, optional
        Name of the column containing the datetime index. If not provided, the index
        of the dataframes are used.

    Returns
    -------
    list: list
        List containing the train and test splits of the input arrays

    """
    output = []
    for array in arrays:
        if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
            if datecol is None:
                dates = array.index.to_series()
            else:
                dates = array[datecol]
            train_index = dates < datetime
            test_index = dates >= datetime
            output.append(array.loc[train_index])
            output.append(array.loc[test_index])
        else:
            raise TypeError("Input must be pandas dataframe or series")

    return output
