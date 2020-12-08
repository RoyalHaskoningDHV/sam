import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_quantile_feature_importances(importances, feature_names=None):
    """
    Create bar graph of feature importances, with highest first.
    Also creates aggregated features over lag features. For this, pass a list of features (sensors)
    as feature_names

    Parameters
    ----------
    importances: pd.DataFrame
        outcome of SamQuantileMLP.quantile_feature_importances()
    feature_names: list or index
        array of features to aggregate for (i.e. lag features were created for this list of input)
    """

    def plot_feature_importances(importances):
        import seaborn as sns
        f = plt.figure(figsize=(10, 3+importances.shape[1]*.2))
        order = list(importances.mean(axis=0).sort_values(ascending=False).index)
        sns.barplot(data=importances, order=order, orient='h')
        sns.despine()
        plt.tight_layout()

    f = plot_feature_importances(importances)

    f_summed = plt.figure()
    if feature_names is not None:
        # and now summed over lag features
        importances_summed = {}
        for sensor in feature_names:
            if not sensor == 'TIME':
                these_cols = [c for c in importances.columns if c.startswith(sensor)]
                importances_summed[sensor] = importances[these_cols].sum(axis=1)
        importances_summed = pd.DataFrame(importances_summed)

        f_summed = plot_feature_importances(importances_summed)

    return f, f_summed
