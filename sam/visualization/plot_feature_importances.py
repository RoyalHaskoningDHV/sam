import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importances(importances, feature_names=None):
    """
    Create bar graph of feature importances, with highest first.
    Also creates aggregated features over lag features. For this, pass a list of features as
    feature_names. It accepts the output of SamQuantileMLP.quantile_feature_importances().
    Alternatively, you can format your own feature importances as a pandas DataFrame with columns
    as features and rows as potentially multiple random iterations.

    Parameters
    ----------
    importances: pd.DataFrame
        with features for columns and potentially multiple random iterations as rows
    feature_names: list or index
        array of features to aggregate for (i.e. lag features were created for this list of input)
        columns should start with this. I.e.: importances for feature_1#lag_1 and feature_1#lag3
        are summed here.

    Returns
    -------
    f: matplotlib.pyplot.Figure
        Bar plot of all features in importances. Error bars indicate variance over iterations
    f_sum: matplotlib.pyplot.Figure
        Bar plot with feature importances summed over lag features.
        Error bars indicate variance over iterations.

    Examples
    --------
    # One way to get to feature importances is to first fit a SamQauntileMLP.
    # In this example, we assumed you did and refer to it as `model`.
    from sam.visualization import plot_quantile_feature_importances
    # note that we need a negative here, as default score function is a loss
    importances = -model.quantile_feature_importances(X, y, sum_time_components=True)
    f, f_sum = plot_quantile_feature_importances(importances,
        list(model.get_input_cols()) + model.time_components)
    """

    def _create_plot(importances):
        import seaborn as sns
        f = plt.figure(figsize=(10, 3+importances.shape[1]*.2))
        order = list(importances.mean(axis=0).sort_values(ascending=False).index)
        sns.barplot(data=importances, order=order, orient='h')
        sns.despine()
        plt.tight_layout()
        return f

    f = _create_plot(importances)

    f_sum = plt.figure()
    if feature_names is not None:
        # and now summed over lag features
        importances_sum = {}
        for feature in feature_names:
            if not feature == 'TIME':
                these_cols = [c for c in importances.columns if c.startswith(feature)]
                importances_sum[feature] = importances[these_cols].sum(axis=1)
        importances_sum = pd.DataFrame(importances_sum)

        f_sum = _create_plot(importances_sum)

    return f, f_sum
