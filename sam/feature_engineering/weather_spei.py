from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class SPEITransformer(BaseEstimator, TransformerMixin):
    """Standardized Precipitation (and Evaporation) Index

    Computation of standardized metric that measures relative drought
    or precipitation shortage.

    SP(E)I is a metric computed per day. Therefore daily weather data
    is required as input. This class assumes that the data contains
    precipitation columns 'RH' and optionally evaporation column 'EV24'.
    These namings are KNMI standards.

    The method computes a rolling average over the precipitation (and
    evaporation). Based on historic data (at least 30 years) the mean
    and standard deviation of the rolling average are computed across years.
    The daily rolling average is then transformed to a Z-score, by dividing
    by the corresponding mean and standard deviation.

    Smoothing can be applied to make the model more robust, and able to
    compute the SP(E)I for leap year days. If ``smoothing=False``, the
    transform method can return NA's

    The resulting score describes how dry the weather is. A very low score
    (smaller than -2) indicates extremely dry weather. A high score (above 2)
    indicates very wet weather.

    See:
    http://www.droogtemonitor.nl/index.php/over-de-droogte-monitor/theorie

    Parameters
    ----------
    metric: {"SPI", "SPEI"}, default="SPI"
        The type of KPI to compute
        "SPI" computes the Standardized Precipitation Index
        "SPEI" computed the Standardized Precipitation Evaporation Index
    window: str or int, default='30D'
        Window size to compute the rolling precipitation or precip-evap sum
    smoothing: boolean, default=True
        Whether to use smoothing on the estimated mean and std for each day of
        the year.
        When ``smoothing=True``, a centered rolling median of five steps is
        applied to the models estimated mean and standard deviations per day.
        The model definition will therefore be more robust.
        Smoothing causes less sensitivity, especially for the std.
        Use the ``plot`` method to visualize the estimated mean and std
    min_years: int, default=30
        Minimum number of years for configuration. When setting less than 30,
        make sure that the estimated model makes sense, using the ``plot``
        method
    model_: dataframe, default=None
        Ignore this variable, this is required to keep the model configured
        when creating a new instance (common in for example cross validation)

    Examples
    ----------
    >>> from sam.data_sources import read_knmi
    >>> from sam.feature_engineering import SPEITransformer
    >>> knmi_data = read_knmi(start_date='1960-01-01', end_date='2020-01-01',
    ...     variables=['RH', 'EV24'], freq='daily').set_index('TIME').dropna()
    >>> knmi_data['RH'] = knmi_data['RH'].divide(10).clip(0)
    >>> knmi_data['EV24'] = knmi_data['EV24'].divide(10)
    >>> spi = SPEITransformer().configure(knmi_data)
    >>> spi.transform(knmi_data)  # doctest: +ELLIPSIS
                SPEI_30D
    TIME ...
    """

    def __init__(
        self,
        metric: str = "SPEI",
        window: str = "30D",
        smoothing: bool = True,
        min_years: int = 30,
        model_: pd.DataFrame = None,
    ):
        self.window = window
        self.metric = metric
        self.smoothing = smoothing
        self.min_years = min_years
        self.model_ = model_

    def _check_configured(self) -> None:
        if self.model_ is None:
            raise NotFittedError("model_ is None, call `configure` first")

    def _check_input(self, X: pd.DataFrame) -> None:
        """Check if required columns are present in dataframe `X`

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to check the columns of
        """
        if "RH" not in X.columns:
            raise ValueError("Dataframe X should contain columns 'RH'")
        if ("EV24" not in X.columns) and (self.metric == "SPEI"):
            raise ValueError("Metric SPEI requires X to have 'EV24' column")

    def _compute_target(self, X: pd.DataFrame) -> pd.Series:
        """Compute target depending on metric type

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to extract target from

        Returns
        -------
        pd.Series
            target series
        """
        if self.metric == "SPI":
            target = X["RH"]
        elif self.metric == "SPEI":
            target = X["RH"] - X["EV24"]
        else:
            raise ValueError("Invalid metric type, choose either 'SPI' or 'SPEI'")
        target = target.rolling(self.window).mean()

        return target

    def configure(self, X: pd.DataFrame, y: Any = None):
        """Fit normal distribution on rolling precipitation (and evaporation)
        Apply this to historic data of precipitation (at least ``min_years`` years)

        Parameters
        ----------
        X: pandas dataframe
            A data frame containing columns 'RH' (and optionally 'EV24')
            and should have a datetimeindex
        y: Any, default=None
            Not used
        """
        self._check_input(X)
        target = self._compute_target(X)

        self._metric_name = self.metric + "_" + self.window
        self._axis_name = "Precip-Evap" if self.metric == "SPEI" else "Precip"

        results = pd.DataFrame(
            {
                self._metric_name: target,
                "month": target.index.month,
                "day": target.index.day,
            },
            index=target.index,
        )

        self.model_ = (
            results.groupby(["month", "day"])[self._metric_name]
            .agg(["count", "mean", "std"])
            .reset_index()
            .sort_values(by=["month", "day"])
        )

        n_years = self.model_["count"].max()
        # Make sure that for at least one day there are ``min_years`` samples
        if n_years < self.min_years:
            raise ValueError(
                f"Provided weather data contains less than"
                f"{self.min_years} years. "
                f"Please provide more data for configuration"
            )

        # Each day should at least have data for 50%
        # of all years in the data, otherwise estimated mean and std
        # are set to nan. This removes leap year days
        self.model_.loc[(self.model_["count"] < (n_years / 2)), ["mean", "std"]] = np.nan
        if self.smoothing:
            # To remove spikes in the mean and std
            # and create a smooth model over the year
            # smoothing is applied. This approach of 5-step median
            # is just a first approach, and does the essential trick
            # Default SP(E)I from literature does not use smoothing
            self.model_["mean"] = (
                self.model_["mean"]
                .rolling(5, center=True, min_periods=1)
                .median(numeric_only=True)
            )
            self.model_["std"] = (
                self.model_["std"].rolling(5, center=True, min_periods=1).median(numeric_only=True)
            )

        return self

    def fit(self, X: pd.DataFrame, y: Any = None):
        """Fit function. Does nothing other than checking input, but is required for a
        transformer. This function wil not change the SP(E)I model. The SP(E)I should be configured
        with the ``configure`` method. In this way, the ``SPEITransfomer`` can be used within a
        sklearn pipeline, without requiring > 30 years of data.

        Parameters
        ----------
        X: pandas dataframe
            A data frame containing columns 'RH' (and optionally 'EV24')
            and should have a datetimeindex
        y: Any, default=None
            Not used
        """
        self._check_configured()
        self._check_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforming new weather data to SP(E)I metric

        Parameters
        ----------
        X : pd.DataFrame
            New weather data to transform

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with single columns
        """
        self._check_configured()
        self._check_input(X)
        target = self._compute_target(X)

        results = pd.DataFrame(
            {
                self._metric_name: target,
                "month": target.index.month,
                "day": target.index.day,
            },
            index=target.index,
        )

        results = results.merge(
            self.model_, left_on=["month", "day"], right_on=["month", "day"], how="left"
        )
        results.index = target.index
        results[self._metric_name] = (results[self._metric_name] - results["mean"]) / results[
            "std"
        ]
        return results[[self._metric_name]]

    def plot(self):
        """Plot model
        Visualization of the configured model. This function shows the
        estimated mean and standard deviation per day of the year.
        """
        self._check_configured()
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        axs[0].plot(self.model_["mean"])
        axs[0].set_ylabel("Mean of " + self._axis_name)
        axs[0].set_xlabel("Day of the year")
        axs[1].plot(self.model_["std"])
        axs[1].set_ylabel("Standard deviation of " + self._axis_name)
        axs[1].set_xlabel("Day of the year")
