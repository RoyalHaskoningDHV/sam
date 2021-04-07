tsfresh
=========

tsfresh is a package that can be used to calculate many timeseries-related features used for analysing time series, especially based on physics, and use them as features in your models. It's pretty straightforward to use, because it has built-in functions that calculate all these features, and select only the relevant ones. To do this, you need the following:

* Some train data, with missing values removed/imputed
* Some train target, to see which features are relevant and which aren't
* Optionally, some validation/test data, to calculate the same features on

tsfresh cannot handle missing values. It is recommended to always be careful with your missing values. For example: ask yourself if missing values contain any extra information, like a sensor failure. If the missing values are mostly random, it is recommended to use the pandas `interpolate` or `fillna` function to interpolate the missing values. Try not to set all the missing values to 0 or -1: this means the time series loses meaning and tsfresh will have worse results!

## Using the tsfresh transformer

The main tsfresh approach looks like this: Given some data `X` and target `y`:
* Keep the original `X` that will later be used to add the created features to
* Make a copy of `X` where we remove unneeded columns and missing values. We will refer to this as `tsdata`.
* Transform `tsdata` to the format that tsfresh requires
* Fit a transformer that calculates all the timefeatures, and adds them to `X`
* Optionally: Use the transformer to also add the same timefeatures to your testdata `X_test`

### Step 0: Obtaining data

For this tutorial, we are going to use synthetic Nereda data. The function `create_synthetic_nereda` is not part of the SAM package, but can be found at https://dev.ynformed.nl/P9 .

If you don't want to use this data, just know that this data contains the specific reactor in the `UnitName` column. One dataframe can contain multiple reactors, whose data is separate and should not be mixed. Also, the time is stored in the `HistBatchStartDate` column. The rest of the columns simply contain data, in numeric format.

```python
X = create_synthetic_nereda(units=['test'], start='2010-01-01', end='2017-05-01')
X = X.dropna(subset=['NH4'])  # drop rows with missing target
y = X['NH4']
```

### Step 1: Preparing tsdata

tsfresh does not support missing data in the `X` dataframe. Furthermore, tsfresh calculation can take quite long, so we want to remove all columns that are not relevant to tsfresh. To do this, we make a copy of `X`, called `X_copy`. Importantly, the output from tsfresh is later added to `X` by looking at the row index. Therefore, we should not change the row index of `X_copy` during this process.

We do not get rid of two columns with meta-data: the column with the timestamps (`HistBatchStartDate`), and the column that specifies which timeseries each row belongs to, in case there are multiple timeseries. If there is only one timeseries in the data, a column will have to be added, for example by saying: `X_copy['UnitName'] = 1`.

```python
# We are careful to not change the row index of tsdata
# Drop colums we don't want tsfresh to calculate on
X_copy = X.drop(['Id', 'HistBatchEndDate'], axis=1)
# We get rid of all missings. Just use simple mean imputation for now.
X_copy = X_copy.fillna(X_copy.mean())
```

### Step 2: Bring into format usable by tsfresh

Tsfresh supports a dataformat where each id corresponds to a single instance. This means that for each instance, the entire rolling window has to be given its own id. For more information about this format, [check the official documentation here](https://tsfresh.readthedocs.io/en/latest/text/forecasting.html). Luckily, there is a convenience function to do exactly this.

```python
from tsfresh.utilities.dataframe_functions import roll_time_series
tsdata = roll_time_series(
    X_copy,  # the normal data where 1 row corresponds to one time measurement
    column_id='UnitName',  # An id that identifies different timeseries
    column_sort='HistBatchStartDate',  # A time id that sorts the data
    column_kind=None,  # optional, for if you want to group certain columns
    rolling_direction=1,  # 1 for rolling windows
    max_timeshift=10  # max rolling window size. To save computation time
)
```

The `max_timeshift` option sets the rolling window size that tsfresh will use to calculate features for a single time point. This should be high enough that tsfresh can calculate interesting features, but the computation time increases quadratically with this parameter, so it should not be set too high.

We can now proceed to calling the transformer described in the previous section. We just have to correctly choose the `column_id` and `column_sort`: in this example: `UnitName` and `HistBatchStartDate`.

### Step 3: Actually calculating the features

To actually calculate all the features, it is recommended to use the sklearn transformer that comes with tsfresh:

```python
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import EfficientFCParameters

transformer = RelevantFeatureAugmenter(
    column_id='UnitName',
    column_sort='HistBatchStartDate',
    timeseries_container=tsdata,
    default_fc_parameters=EfficientFCParameters()
)

# transformer.set_params(timeseries_container=tsdata)  # optional
X_new = transformer.fit_transform(X, y)
```

The result will be that `X_new` will have the same columns as `X`, but with extra columns added, as calculated by tsfresh. This sklearn transformer can be used in combination with other sklearn transformers, such as pipelines and column transformers. You will just have to be careful to calculate `tsdata` beforehand, and make sure that it corresponds to `X`.

### Step 4: Applying the transformer to test data

After all that, to apply this transformation to the test data (to make a prediction), we need to do this same process, except with `transform` instead of `fit_transform`. Here we calculate some `testtsdata`, which has had the same preprocessing as `tsdata`.

Imagine we have some data `X_test` with the same format as `X`:

```python
# The same preprocessing as was done to tsdata, but now for the test data instead
X_test_copy = X_test.drop(['Id', 'HistBatchEndDate'], axis=1)
X_test_copy = X_test_copy.fillna(X_test_copy.mean())
tstestdata = roll_time_series(X_test_copy, ...)  # same parameters as earlier

# apply the transformer to the test data
transformer.set_params(timeseries_container=tstestdata)
X_test_new = transformer.transform(X_test)
```

This will only calculate the features that were previously selected, and add them all to `X_test`.

## More details about tsfresh

### fc_parameters

In the transformer, there was the variable `default_fc_parameters`. There are three possible values for `default_fc_parameters`: 
* ComprehensiveFCParameters()
* EfficientFCParameters()
* MinimalFCParameters()

The only difference is in how many features they compute: Comprehensive computes all features available in tsfresh, but will take a long time to compute. Efficient computes all but the most computationally intensive. Minimal only computes a few. 

Minimal is only for testing purposes, beause it only adds simple features like the mean, median, maximum, etcetera. However, because tsfresh can take very long if you have more than a few thousand rows, it is recommended to always start with Minimal! That way, you can see if your pipeline works, before you step up to Efficient or Comprehensive and find out after a long computation that you had a bug in your code.

### Feature selection

tsfresh computes many, many features, in the order of hundreds. Furthermore, it computes those hundreds of features for each input feature. For example, if `tsdata` contains 7 features, it will compute 7 sets of hundreds of features. The output will therefore contain thousands of columns. This is far too much, and most of these features will be completely irrelevant. tsfresh already filters out the irrelevant features by using `RelevantFeatureAugmenter` (Hence the word `relevant`). It does this by comparing each of the thousands of features to the target variable, and removing those that have absolutely no relation.

Therefore, the transformer will not return thousands of features, but perhaps only a few dozen. There is no hard rule on this. There are multiple significance tests applied, depending on the type of feature: either the Fisher test, the Mann-Whitney test, or the Kendall test. Only the features with the lowest p-values are kept, with the exact threshold decided by the Benjamini Hochberg procedure. Therefore, the number of returned features can differ depending on the data: if you have purely random noise data, there might be no new features returned at all.

Furthermore, because of the complexity of this entire process, it is not guaranteed that running the code with 99% similar data will return 99% similar feature columns. However, we haven't done enough experimentation to know exactly how much this varies.

### Obtaining specific features

Knowing that tsfresh is not guaranteed to return the same features, we might want to ask it to calculate specific features, instead of leaving it all up to the selection algorithm described above. There are two ways of doing this:

* The first, and probably easiest way is when you just want to use the same columns as were selected in a previous run. This can be done by using the sklearn-transformer as described in the first section. By using `transform`, only the features selected during the `fit` will be calculated. This means the `transform` call will be much faster than the `fit` call, since only a few features are calculated, instead of hundreds.

* The second method applies if you want even more fine control over which features are calculated. In this case, [check the tsfresh documentation here](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html). This page describes creating a dictionary that lists all the features you want to calculate. This is obviously more labor-intensive. Although it is not mentioned on this page, this approach can also be combined with the sklearn-transformer: just pass the dictionary to the transformer, in the `default_fc_parameters` argument.

### Troubleshooting: ZeroDivisionError

During development, we encountered the following error: `ZeroDivisionError: integer division or modulo by zero`. We have not (yet) found out exactly what triggers this error, but we know how to fix it.

The key is the `time` column. When this column is converted to a simple range from 1 to `n`, the problem dissapears. This code is ran in the 'Step 1: Preparing tsdata' section.

```python
X_copy['HistBatchStartDate'] = range(X_copy.shape[0])
```

This should not change the result, because tsfresh does not actually use the content of the `time` column: it's only used for sorting the data correctly. It is important though, to double check that this does not mess up the sorting. To be safe, you should always check if your dataframe is already sorted by the `time` column. If it's not, the above operation could destroy the meaning of your time column!
