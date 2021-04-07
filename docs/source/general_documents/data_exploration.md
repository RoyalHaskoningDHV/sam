Data exploration
================

There are many ways to do a first data exploration, this document is to help with getting started and give a few guidelines for explorering time series. A nice start is probably the [the SAM exploration tool](#sam-exploration-tool)

In addition to normal exploration, missing values, high cardinality, etc. time series have a few unique characteristics.

## General pitfalls

### Frequency
How often do you get measurements? Is the frequency varying in one source? What about between sources? Do the measurements shift over time? You might want to convert them to a fixed frequency.

### Gaps
Are rows 'missing' when you look at the timestamps of the data? How often does this happen? If data misses, is this just in one sensor-series or in a group of sensors or _all_ series at the same time? It might be useful to 'complete' your time series, make it a fixed frequency with NA values if there are gaps. In SAM there is the `complete_timestamps` function.

### Time zones
In what time zone was the data recorded and what time zone does your system use? Especially important when joining multiple sources. Sometimes the time zone information is recorded in the date field (something like TZ), often it is not, and you should contact the person providing the data.

### Winter/Summertime
Does your time period span summer/wintertime changes? In this case you get ambiguous data, two measurements at the same time, or big gaps. Prefer to work with timestamps (for example, the unix epoch timestamps: seconds since 1970-01-01, often the default), this removes the problem. If you get ambiguous data, you might be able to determine the correct order from the order of lines in the file. 

## Tools and packages to use

### Python

With a pandas dataframe you can use [pandas profiling](https://github.com/pandas-profiling/pandas-profiling). This package will generate a report on _all_ your columns. This can be a quick way to get an overview, but of course you need to take some time to interpret this document.

A package specific for visualising information about missing values is [missingno](https://github.com/ResidentMario/missingno).

### R
While the SAM-package is a python package, you can sometimes do a first exploration faster in R. For inspiration you can read (and copy code) from [Spectral Analysis of Time Series](https://rstudio-pubs-static.s3.amazonaws.com/9428_1197bd003ebd43c49b429f22ea4f36e5.html) A few packages that can be used for this:

#### [tsstats](http://rpubs.com/sinhrks/plot_tsstats)
The base package `stats` provides a lot of useful functions, this package is a wrapper to make them easily available in ggplot. 

#### [naniar](https://cran.r-project.org/web/packages/naniar/vignettes/getting-started-w-naniar.html)
Provides insight into missing values and is more capable than Amelia. Especially useful for timeseries is the `miss_var_run()` function. This gives a table with the consecutive runs of missing and non-missing data. For this to work properly you first must make your dataframe 'complete', make sure the frequency is static.

#### [datamaid](https://cran.r-project.org/web/packages/dataMaid/index.html)
Generates a rapport like pandas profiling.

#### [radiant](https://vnijs.github.io/radiant/)
An interactive tool, more like BI so probably not that useful. Can also do some machine learning.

## SAM exploration tool
To speed up inital analysis and help during conversations with domain exports at the client we created a simple (shiny) tool. It takes data from the database in the development environment (see [Production platform](production_platform.html) for more components) in the long format as described in [the data storage section](data_storage.html). Right now it only takes data from a mongodb database. You would use the tool as follows:

1. Process the data to long-format and store it as a csv, rds, or feather file.
2. Load op the SAM-exploration tool: [Currenty on app-test](https://app-test.ynformed.nl/app/samexploration)
3. Use the various views for insights into correlations, summary statistics etc.

### Tips

- After loading the datamaid rapport the app crashes, simply refresh the page to make it work again
- Filters and selections only affect plots next to them, except for the global date filter in the side menu, this affects all
