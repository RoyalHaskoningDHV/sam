Data storage
=============

To make it easier to work together in a project and with your future self we defined a standard way of storing data. This way was designed to be easy to use and also support the different programming languages. Right now we use two main formats:


## Formats
1. Long format

`ID, TIME, TYPE, VALUE`

Just four columns and lots of rows. Specifically for sensor measurements, probably one of the first ways you'll save your data. Usefull in the [SAM exploration tool](data_exploration.html#sam-exploration-tool) This is a long format, where:

- ID specificies the object (a reactor, pump etc.)
- TIME something like a date
- TYPE what is measured, like NH4 or millimeters rain
- VALUE the measured sensor value.

2. Wide format

`ID, TIME, feature_1, ..., feature_n`
This format is mainly used for feature tables, and in your pipeline. A more traditional way and probably the last final format of your data. 

A few of SAM's functions create or use grouped features in a wide format. They group based on column name `A#B`, where `A` is the group name and `B` the feature name. Note that a feature named `A#B#C` will have group `A` and feature name `B#C`.

## Locations
Currently we store data in mongodb and influxdb, with the latter having the preference if you want to easily visualise in grafana.

## Considerations

The long format might make some operations counter-intuitive, and when it comes to effeciency it can often be faster to select a specific column than filter all rows. However, the source data might have varying time intervals in different sources, and this is easier to represent and fix in a long format.

The way of storing data in data science projects is usually less important, as you often just load all your data (or a time range) anyway. For this reason most databases will have a similar performance. We chose mongodb as it offers schema flexibility, influxdb as it is timeseries specific and integrates well with grafana. Finally we have a postgress database as an easy to use SQL database. If you often access only parts of your data you could use a more project-specific way of storing the data (parquet fikes seem the hip and cool way nowadays).

Other considerations can be found [in this task on phabricator](https://dev.ynformed.nl/T155).
