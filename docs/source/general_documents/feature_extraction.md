Feature Extraction
==================

This is the start of the documentation on which features to use when.

## Transforms
Best practices on types of transforms that you can apply.

### Summarizing
Summarizing a window prior to the prediction moment with some basic functions is always a good starting point:
- Basic: min, max, median, sum, number of positive values
- Distribution: mean, std, var, skewness, kurtosis
- Miscellaneous: crossing points (the number of times the mean is crossed)

Forecasting techniques traditionally used for time series can also be used as features to summarize this window, eg: 
- Weighted average
- Exponential smoothing (double/triple)
- Holt-Winters
- ...

### Deviation
Next to using a metric itself, we can use several measures of deviations between a metric and its expected value (based on recent historical data). Historically relative properties can be more important than the properties itself and can sometimes capture physical properties.

Such deviations should be calculated if:
- We expect (relatively) constant values.
- Especially useful for monitoring the behaviour of pumps.

### Autocorrelation / lag
Calculated lagged features can help model autocorrelation. However when using these, there is also a risk of just predicting previous values when the changes in the response variable are mostly gradual over time. Because of this, accuracy metrics can be very misleading, especially when we are looking for anomalies.

To test the predictive powers in case of autocorrelation, you can define the model to predict the difference in values between time steps, rather than the value itself. 

Using lagged features is:
- Not recommended when the focus is on anomalies and understanding the prediction.
- Recommended when the focus is on predicting as precise as possible.

### Fourier transform (FFT)
Decompose the signal into frequencies. The output we use is a column with an amplitude, for each possible frequency. So the higher column values, the more present this frequency is. More information on [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform).

Fourier transforms are especially useful in case:
- We predict *whether* something will happen, rather than *when*.
- We classify a time series or look for the presence of specific patterns.
- There is strong periodicity in the data.

Extension: you can also add the positions of the peaks in the frequency spectrum as features. This adds information on which frequency is most important.

### Wavelet transform (WT)
Wavelet transforms are similar to fourier transforms, however the key difference is the temporal resolution: a wavelet captures both frequency *and* location information. This not only gives us the frequencies that are present, but also at which time these frequencies occur.

Wavelets are better suited in case:
- We want to capture how the frequencies change over time
- Abrupt changes in frequency for a short period of time (peaks) should influence our prediction heavily.

The same extension as in fourier transforms holds for wavelets. You can add additional features on the locations of peaks.

### Symbolic Aggregate Approximation (SAX)
Discretization of time series into one symbolic string. First we convert the time series to PAA (Piecewise Aggregate Approximation) representation and then we convert to symbols. This means that we can summarize an entire time series (or window of a time series) into one value. See an [example](https://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html).

Advantages:
- Only one value instead of multiple columns
- Good for visualisation
- Useful when comparing/finding patterns in a time series

### Miscellaneous: tsfresh
The python library [tsfresh](https://github.com/blue-yonder/tsfresh) calculates a comprehensive number (approximately 65) of features for time series (Feature Calculation). The calculated features are very diverse, ranging from 'time reversal asymmetric statistic' to 'entropy', etc. There is some overlap with the summarizing functions in this document. It also contains automatic feature selection (Feature Filtering) based on significant correlation with the target. 
Since many of the features are not very intuitive, it is difficult to know which ones to use in a specific project. However there might be very powerful features in there. It is therefore recommended to use the Feature Calculation *and* Feature Filtering of tsfresh. By using the filtering, the significance of the features is tested and only the most important ones should be added to the model. These features should then be added next to other mentioned features in this document and should not be used as a replacement. This way we keep the intuitive features which might not be significant based only on the target variable, but have an interaction effect with other variables. 


## Type of variables
Best practices for types of variables that you might have in your model.

### Seasonality
There are many ways to capture seasonality, to capture the cyclic nature, using more than one might be the best option. Which one(s) to use depends also on what model you use, for example some features might work better for tree-based models and others for linear models. In tree-based models we try to group similar time periods together by giving them a similar value for the feature.
The following uses months as example, but applies on different levels, like: seasons, months, weeks, days, hours, etc.

- Categorical encoding (or dummy encoding in Python)
    - When: All possible groupings are possible.
    - How: A column with the name of the month is added.
- Sinus encoding
    - When: Months that are close to each other are easily grouped. 
    - How: We add a column with the discrete values of a sinus corresponding to the months. The frequency depends on the level of seasonality we want to capture (for yearly patterns like seasons and months the frequency is 1, for weekly patterns the frequency is 52, etc.)
- Target encoding
    - When: Months that have a similar value in the target are easily grouped.
    - How: Each month gets the average value of the target of all months. 
- Numeric encoding
    - When: Months that are close to each other are very easily grouped, however the cyclic nature isn't accounted for.
    - How: To partly account for the cyclic nature, multiple columns can be added, eg 1: jan till 12: dec, but also 1: jul till 12: jun.

*Another option would be to first remove the seasonality and not include any seasonal features in the model.*

The year should be added as a numeric variable if we suspect a trend in the data, however depending on whether a model is chosen that can extrapolate values, the trend should be removed from the data separately by decomposition.

Holidays should be included as a categorical or boolean variable. For this we need a standardised list of holidays.

### Pumps
When working with pumps (or other discrete processes that take some time), some features that can be used are:
- Features based on the number of pumping events per day.
- The pump's flow rate.
- The duration of the pumping events.
- The ratio of a pump's flow rate to amount of handle motion (i.e volume of water per human effort).

For all these categories, using the deviations as well as the values itself, also captures the physical properties of the pumps and is highly recommended.

### Water level
Since the water level is fairly constant or changing gradually, there is often high autocorrelation present. Using this in our features in most cases means getting a better performance, but might also lead to missing sensor defects/degradation. Also see the section on autocorrelation. 

