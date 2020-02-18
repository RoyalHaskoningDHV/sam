import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sam.metrics import train_mean_r2


def performance_evaluation_fixed_predict_ahead(y_true_train, y_hat_train, y_true_test, y_hat_test,
                                               resolutions=[None], predict_ahead=0):
    """
    This function evaluates model performance over time for a single given predict ahead.
    It plots and returns r-squared, and creates a scatter plot of prediction vs true.
    It does so for different temporal resolutions and for both test and train sets.
    The idea of evaluating performance for different time resolutions,
    is that this gives insight into the resolution of the underlying patterns in your data
    that the model is capturing (e.g. if you have very high time-resolution data, the model
    might not capture every minute-to-minute change, but it could capture the hourly or daily
    patterns). Using this approach, you can find out what the best time-resolution is to use
    in for instance the sam.visualizations.quantile_plot.sam_quantile_plot.

    Parameters
    ----------
    y_true_train: pd.Series
        Series that contain the true train values.
    y_hat_train: pd.DataFrame
        DataFrame that contain the predicted train values (output of SamQuantileMLP model.predict)
    y_true_test: pd.Series
        Series that contain the true test values.
    y_hat_test: pd.DataFrame
        DataFrame that contain the predicted test values (output of SamQuantileMLP model.predict)
    resolutions: list (default=[None])
        List of strings (and/or None) that are interpretable by pandas resampler.
        If set to None, will return results for the native data resolution.
        Valid options are e.g.: [None], [None, '15min', '1H'], or ['1H', '1D']
    predict_ahead: int (default=0)
        predict_ahead to display performance for

    Returns
    ------
    r2_df: pd.DataFrame
        Dataframe that contains the r2 values per test/train set and resolution combination.
        columns: ['R2', 'dataset', 'resolution']
    bar_fig: matplotlib figure
        Figure object that displays these r-squareds.
    scatter_fig: matplotlib figure
        Figure object that displays predicted vs true data for the different resolutions.
    best_res: string
        The resolution with the maximum R2 in the train set.

    Example
    -------

    # assuming you have some y_true_train, y_true_test and predictions y_hat_train and y_hat_test:
    from sam.visualization import performance_evaluation_fixed_predict_ahead
    r2_df, bar_fig, scatter_fig, best_res = performance_evaluation_fixed_predict_ahead(
        y_true_train,
        y_hat_train,
        y_true_test,
        y_hat_test,
        resolutions=[None, '15min', '1H', '3H', '6H', '1D'])

    # display the results
    bar_fig.show()
    scatter_fig.show()
    print('best resolution found at %s'%best_res)
    r2_df.head()

    # print some results
    best_res_r2 = r2_df.loc[(r2_df['dataset']=='train') &
                            (r2_df['resolution'] == best_res), 'R2'].values[0]
    native_r2 = r2_df.loc[(r2_df['dataset']=='train') &
                          (r2_df['resolution'] == 'native'), 'R2'].values[0]
    print('best resolution found at %s (%.3f vs %.3f native)'%(best_res, best_res_r2, native_r2))
    """

    import seaborn as sns

    # initialize scatter figure
    scatter_fig = plt.figure(figsize=(len(resolutions)*3, 6))

    # select and shift the requested predict ahead
    y_hat_train = y_hat_train['predict_lead_%d_mean' % predict_ahead].shift(predict_ahead)
    y_hat_test = y_hat_test['predict_lead_%d_mean' % predict_ahead].shift(predict_ahead)

    # compute the r-squared (r2) for the different temporal resolutions, for train and test data
    r2_list, dataset_list, resolution_list = [], [], []
    for ri, res in enumerate(resolutions):

        # resample data to desired resolution of requested
        if res is not None:
            res_label = res
            y_true_train_res = y_true_train.resample(res).mean()
            y_hat_train_res = y_hat_train.resample(res).mean()
            y_true_test_res = y_true_test.resample(res).mean()
            y_hat_test_res = y_hat_test.resample(res).mean()
        else:
            res_label = 'native'
            y_true_train_res = y_true_train
            y_hat_train_res = y_hat_train
            y_true_test_res = y_true_test
            y_hat_test_res = y_hat_test

        # compute r2 with custom r2 function (in sam.metrics)
        test_r2 = train_mean_r2(y_true_test_res, y_hat_test_res, np.mean(y_true_train_res))
        train_r2 = train_mean_r2(y_true_train_res, y_hat_train_res, np.mean(y_true_train_res))

        # append results to lists
        r2_list.append(train_r2*100)
        dataset_list.append('train')
        resolution_list.append(res_label)
        r2_list.append(test_r2*100)
        dataset_list.append('test')
        resolution_list.append(res_label)

        # create scatter plot of train results:
        alpha = np.min([1000/len(y_true_train_res), 1])
        plt.subplot(2, len(resolutions), ri+1)
        ymin = np.min([y_true_train.min(), y_hat_train.min()])
        ymax = np.max([y_true_train.max(), y_hat_train.max()])
        plt.plot([ymin, ymax], [ymin, ymax], c='gray', ls='--')
        plt.plot(y_true_train_res.values, y_hat_train_res.values, 'o', alpha=alpha)
        plt.title('train ' + res_label)
        plt.xlim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if ri > 0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel('true')
            plt.ylabel('predicted')

        # create scatter plot of test results:
        plt.subplot(2, len(resolutions), ri+1 + len(resolutions))
        plt.plot([ymin, ymax], [ymin, ymax], c='gray', ls='--')
        plt.plot(y_true_test_res.values, y_hat_test_res.values, 'o', alpha=alpha, color='orange')
        plt.title('test ' + res_label)
        plt.xlim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if ri > 0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel('true')
            plt.ylabel('predicted')

    # options for scatter plot
    sns.despine()
    plt.tight_layout()

    # create bar plot of different r-squareds
    r2_df = pd.DataFrame({'R2': r2_list, 'dataset': dataset_list, 'resolution': resolution_list})
    bar_fig = plt.figure(figsize=(6, 4))
    plt.axhline(0, c='k')
    sns.barplot(data=r2_df, x='resolution', y='R2', hue='dataset')
    plt.ylabel('Variance Explained (%)')
    sns.despine()
    plt.ylim(0, 100)

    # calculate best resolution as the maximum resolution R2 in the train set
    best_res = r2_df.iloc[r2_df.loc[r2_df['dataset'] == 'train', 'R2'].idxmax()]['resolution']

    return r2_df, bar_fig, scatter_fig, best_res
