from typing import Union
import pandas as pd


def datetime_train_test_split(
    *arrays: Union[pd.DataFrame, pd.Series],
    datetime: str,
):
    """
    Split the dataframe into train and test sets based on datetime index values

    Parameters
    ----------
    arrays : Union[pd.DataFrame, pd.Series]
        Allowed inputs are pandas dataframes or series (with datetime index)
    datetime : str
        Datetime to split the dataframe on

    Returns
    -------
    list: list
        List containing the train and test splits of the input arrays

    """
    output = []
    for array in arrays:
        if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
            output.append(array.loc[:datetime])
            output.append(array.loc[datetime:])
        else:
            raise TypeError("Input must be pandas dataframe or series")

    return output
