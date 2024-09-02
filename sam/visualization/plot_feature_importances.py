from typing import Iterable

import pandas as pd


def plot_feature_importances(importances: pd.DataFrame, feature_names: Iterable = None):
    """
    Create bar graph of feature importances, with highest first.
    Also creates aggregated features over lag features. For this, pass a list of features as
    feature_names. It accepts the output of MLPTimeseriesRegressor.quantile_feature_importances().
    Alternatively, you can format your own feature importances as a pandas DataFrame with columns
    as features and rows as potentially multiple random iterations.

    Parameters
    ----------
    importances: pd.DataFrame
        Dataframe with features as columns and potentially multiple random iterations as rows.
    feature_names: iterable of strings or None (default=None)
        Iterable of column names (or starting column names) to aggregate for. Every element of
        feature_names is the common start name of each feature, i.e.: importances for
        feature_1#lag_1 and feature_1#lag3 are summed.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Bar plot of all features in importances. Error bars indicate variance over iterations
    fig_sum: matplotlib.pyplot.Figure
        Bar plot with feature importances summed over lag features.
        Error bars indicate variance over iterations.

    Examples
    --------
    >>> # One way to get to feature importances is to first fit a MLPTimeseriesRegressor.
    >>> # In this example, we assumed you did and refer to it as `model`.
    >>> from sam.visualization import plot_feature_importances
    >>> # note that we need a negative here, as default score function is a loss
    >>> # Example with a fictional dataset with only 2 features
    >>> import pandas as pd
    >>> import seaborn
    >>> from sam.models import MLPTimeseriesRegressor
    >>> from sam.feature_engineering import SimpleFeatureEngineer
    >>> from sam.datasets import load_rainbow_beach
    ...
    >>> data = load_rainbow_beach()
    >>> X, y = data, data["water_temperature"]
    >>> test_size = int(X.shape[0] * 0.33)
    >>> train_size = X.shape[0] - test_size
    >>> X_train, y_train = X.iloc[:train_size, :], y[:train_size]
    >>> X_test, y_test = X.iloc[train_size:, :], y[train_size:]
    ...
    >>> simple_features = SimpleFeatureEngineer(
    ...     rolling_features=[
    ...         ("wave_height", "mean", 24),
    ...         ("wave_height", "mean", 12),
    ...     ],
    ...     time_features=[
    ...         ("hour_of_day", "cyclical"),
    ...     ],
    ...     keep_original=False,
    ... )
    ...
    >>> model = MLPTimeseriesRegressor(
    ...     predict_ahead=(0,),
    ...     feature_engineer=simple_features,
    ...     verbose=0,
    ... )
    ...
    >>> model.fit(X_train, y_train)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    >>> score_decreases = model.quantile_feature_importances(
    ...     X_test[:100], y_test[:100], n_iter=3, random_state=42)
    >>> importances = -model.quantile_feature_importances(X, y, random_state=42)
    >>> fig, fig_sum = plot_feature_importances(importances,
    ...     list(model.get_input_cols()) + model.get_feature_names_out())
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _create_plot(importances):
        f = plt.figure(figsize=(10, 3 + importances.shape[1] * 0.2))
        order = list(importances.mean(axis=0).sort_values(ascending=False).index)
        sns.barplot(data=importances, order=order, orient="h")
        sns.despine()
        plt.tight_layout()
        return f

    fig = _create_plot(importances)

    fig_sum = plt.figure()
    if feature_names is not None:
        # and now summed over lag features
        importances_sum = {}
        for feature in feature_names:
            if feature != "TIME":
                these_cols = [c for c in importances.columns if c.startswith(feature)]
                importances_sum[feature] = importances[these_cols].sum(axis=1)
        importances_sum = pd.DataFrame(importances_sum)

        fig_sum = _create_plot(importances_sum)

    return fig, fig_sum
