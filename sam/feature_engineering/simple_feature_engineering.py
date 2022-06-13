from typing import List, Tuple, Union, Optional

import numpy as np
import pytz
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


COMPONENT_FUNCTION = {
    "second_of_minute": lambda x: x.dt.second,
    "second_of_hour": lambda x: x.dt.second + x.dt.minute * 60,
    "second_of_day": lambda x: x.dt.second + x.dt.minute * 60 + x.dt.hour * 3600,
    "minute_of_hour": lambda x: x.dt.minute,
    "minute_of_day": lambda x: x.dt.minute + x.dt.hour * 60,
    "hour_of_day": lambda x: x.dt.hour,
    "hour_of_week": lambda x: x.dt.hour + x.dt.weekday * 24,
    "day_of_week": lambda x: x.dt.isocalendar().day,
    "day_of_month": lambda x: x.dt.day,
    "day_of_year": lambda x: x.dt.dayofyear,
    "week_of_year": lambda x: x.dt.isocalendar().week,
    "month_of_year": lambda x: x.dt.month,
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
        The column is the name of the column to be transformed, the method is the
        method to be used (string), and the window is the window size (integer or string).
        Valid methods are "lag" or any of the pandas rolling methods (e.g. "mean", "median", etc.).
    time_features : list (default=[])
        List of tuples of the form (component, type). Can also be provided as a
        dataframe with columns ['component', 'type'].
        Supported components are:
            - "second_of_minute"
            - "second_of_hour"
            - "seconds_of_day"
            - "minute_of_hour"
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
    timezone: str, optional (default=None)
        if tz is not None, convert the time to the specified timezone, before creating features.
        timezone can be any string that is recognized by pytz, for example `Europe/Amsterdam`.
        We assume that the TIME column is always in UTC,
        even if the datetime object has no tz info.
    drop_first : bool (default=True)
        Whether to drop the first value of time components (used for onehot encoding)
    keep_original : bool (default=False)
        Whether to keep the original columns in the dataframe.

    Example
    -------
    >>> from sam.feature_engineering import SimpleFeatureEngineer
    >>> from sam.data_source import read_knmi
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    ...                  variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    ...                             'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T'].shift(-1)
    >>> X = data.drop('T', axis=1)
    >>> fe = SimpleFeatureEngineer(rolling_features=[('T', 'mean', 12), ('T', 'mean', '24')], 
    ...                            time_features=[('hour_of_week', 'onehot')])
    >>> X_fe = fe.fit_transform(X, y)
    >>> X_fe.head()
    """

    def __init__(
        self,
        rolling_features: Optional[Union[List[Tuple], pd.DataFrame]] = [],
        time_features: Optional[Union[List[Tuple], pd.DataFrame]] = [],
        time_col: Optional[str] = None,
        timezone: Optional[str] = None,
        drop_first: bool = True,
        keep_original: bool = False,
    ) -> None:
        super().__init__()
        self.rolling_features = self._input_df_to_list(rolling_features)
        self.time_features = self._input_df_to_list(time_features)
        self.time_col = time_col
        self.timezone = timezone
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
            raise ValueError(f"Invalid data type: {type(data)}, provide a list or dataframe")

    def _fix_timezone(self, datetime):
        """Change timezone before calculating components."""
        if self.timezone is not None:
            if datetime.dt.tz is not None:
                if datetime.dt.tz != pytz.utc:
                    raise ValueError(
                        "Data should either be in UTC timezone or it should have no"
                        " timezone information (assumed to be in UTC)"
                    )
            else:
                datetime = datetime.dt.tz_localize("UTC")
            datetime = datetime.dt.tz_convert(self.timezone)
        return datetime

    def _get_time_column(self, X, component):
        # First select the datetime column
        if self.time_col:
            datetime = X[self.time_col]
        else:
            datetime = X.index.to_series().copy()        

        # Fix timezone
        datetime = self._fix_timezone(datetime)

        # Then select the component
        if component in COMPONENT_FUNCTION:
            return COMPONENT_FUNCTION[component](datetime)
        else:
            raise ValueError(f"Invalid component: {component}")

    def feature_engineer_(self, X, y=None):
        if self.keep_original:
            X_out = X.copy()
        else:
            X_out = pd.DataFrame(index=X.index, columns=[])

        # Rolling features
        for col, method, window in self.rolling_features:
            colname = f"{col}_{method}_{window}"
            if method == "lag":
                X_out[colname] = X[col].shift(window)
            else:
                X_out[colname] = X[col].rolling(window=window).agg(method)

        # Time features
        for component, type in self.time_features:
            colname = f"{component}_{type}"
            comp_min, comp_max = COMPONENT_RANGE[component]

            if type == "onehot":
                # we do not make a dummy of the last value because of collinearity
                if self.drop_first:
                    comp_min += 1
                for value in range(comp_min, comp_max + 1):
                    comp_series = self._get_time_column(X, component)
                    colname_ = f"{colname}_{value}"
                    X_out[colname_] = (comp_series == value).astype(int)

            elif type == "cyclical":
                comp_series = self._get_time_column(X, component)
                # scale to 0,1, then to 0,2pi and then to -1,1
                comp_norm = (comp_series - comp_min) / (comp_max - comp_min + 1)
                X_out[colname + "_sin"] = np.sin(2 * np.pi * comp_norm)
                X_out[colname + "_cos"] = np.cos(2 * np.pi * comp_norm)
            else:
                raise ValueError(f"Invalid type: {type}")

        return X_out
