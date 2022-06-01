from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sam.feature_engineering import BaseFeatureEngineer


COMPONENT_RANGE = {
    "second_of_minute": (0, 59),
    "second_of_hour": (0, 3599),
    "second_of_day": (0, 86399),
    "minute_of_hour": (0, 59),
    "minute_of_day": (0, 1439),
    "hour_of_day": (0, 23),
    "hour_of_week": (0, 167),
    "day_of_week": (1, 7),
    "day_of_month": (1, 31),
    "day_of_year": (1, 366),
    "week_of_year": (1, 53),
    "month_of_year": (1, 12),
}


class SimpleFeatureEngineer(BaseFeatureEngineer):
    """
    Base class for simple time series feature engineering. Provides a method to
    create two types of features: rolling features and time components (one hot or cyclical).

    Parameters
    ----------
    rolling_features : list or pandas.DataFrame (default=[])
        List of tuples of the form (column, method, window). Can also be provided as a
        dataframe with columns: ['column', 'method', 'window'].
        Valid methods are "lag" or any of the pandas rolling methods (e.g. "mean", "median", etc.).
    time_features : list (default=[])
        List of tuples of the form (component, type). Can also be provided as a
        dataframe with columns ['component', 'type'].
        Supported components are:
            - "seconds_of_day"
            - "minute_of_day"
            - "hour_of_day"
            - "hour_of_week"
            - "day_of_week"
            - "day_of_month"
            - "day_of_year"
            - "week_of_year"
            - "month_of_year"
        Valid types are:
            - "onehot"
            - "cyclical"
    time_col : str (default=None)
        Name of the time column (e.g. "TIME"). If None, the index of the dataframe is used.
    drop_first : bool (default=True)
        Whether to drop the first value of time components (used for onehot encoding)
    keep_original : bool (default=False)
        Whether to keep the original columns in the dataframe.
    """

    def __init__(
        self,
        rolling_features: Union[List[Tuple], pd.DataFrame] = [],
        time_features: Union[List[Tuple], pd.DataFrame] = [],
        time_col: str = None,
        drop_first: bool = True,
        keep_original: bool = False,
    ) -> None:
        super().__init__()
        self.rolling_features = self._input_df_to_list(rolling_features)
        self.time_features = self._input_df_to_list(time_features)
        self.time_col = time_col
        self.drop_first = drop_first
        self.keep_original = keep_original

    @staticmethod
    def _input_df_to_list(
        data: pd.DataFrame,
    ) -> List[Tuple]:
        if isinstance(data, pd.DataFrame):
            return list(data.to_records(index=False))
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(
                f"Invalid data type: {type(data)}, provide a list or dataframe"
            )

    def _get_time_column(self, X, component):
        # First select the datetime column
        if self.time_col is None:
            datetime = X.index.to_series().copy()
        else:
            datetime = X[self.time_col]

        # Then select the component
        if component in ["second_of_minute", "second"]:
            return datetime.dt.second
        elif component == "second_of_hour":
            return datetime.dt.second + datetime.dt.minute * 60
        elif component in ["seconds_of_day", "secondofday"]:
            return (
                datetime.dt.second + datetime.dt.minute * 60 + datetime.dt.hour * 3600
            )
        elif component == "minute_of_hour":
            return datetime.dt.minute
        elif component == "minute_of_day":
            return datetime.dt.minute + datetime.dt.hour * 60
        elif component == "hour_of_day":
            return datetime.dt.hour
        elif component == "hour_of_week":
            return datetime.dt.hour + datetime.dt.weekday * 24
        elif component == "day_of_week":
            return datetime.dt.isocalendar().day
        elif component == "day_of_month":
            return datetime.dt.day
        elif component == "day_of_year":
            return datetime.dt.dayofyear
        elif component == "week_of_year":
            return datetime.dt.isocalendar().week
        elif component == "month_of_year":
            return datetime.dt.month
        else:
            raise NotImplementedError(f"Component {component} not implemented.")

    def feature_engineer(self, X):
        if self.keep_original:
            X_out = X.copy()
        else:
            X_out = pd.DataFrame(index=X.index, columns=[])

        # Rolling features
        for feature in self.rolling_features:
            col, method, window = feature
            colname = f"{col}_{method}_{window}"
            if method == "lag":
                X_out[colname] = X[col].shift(window)
            else:
                X_out[colname] = X[col].rolling(window=window).agg(method)

        # Time features
        for feature in self.time_features:
            component, type = feature
            colname = f"{component}_{type}"
            comp_min, comp_max = COMPONENT_RANGE[component]

            if type == "onehot":
                # we do not make a dummy of the last value because of collinearity
                for value in range(comp_min, comp_max):
                    comp_series = self._get_time_column(X, component)
                    colname_ = f"{colname}_{value}"
                    X_out[colname_] = (comp_series == value).astype(int)

            elif type == "cyclical":
                comp_series = self._get_time_column(X, component)
                # Add 1 so that start/end are not the same
                comp_norm = (comp_series - comp_min) / (comp_max - comp_min + 1)
                X_out[colname + "_sin"] = np.sin(2 * np.pi * comp_norm)
                X_out[colname + "_cos"] = np.cos(2 * np.pi * comp_norm)

            else:
                raise ValueError(f"Invalid type: {type}")

        return X_out
