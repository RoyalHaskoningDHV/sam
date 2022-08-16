Data formats
============

To make it easier to work together in a project we defined a standard way of storing data.
This way was designed to be easy to use and also support the different programming languages.
To reshape data from one format to the other, see :ref:`sam-format-reshaping-functions`.

Right now we use two main formats:

Long format (Sam data)
----------------------

A lot of functions work on long format data (usually called sam-data). This is the easiest way to store data, since it doesn't require a lot of columns and is more sparse when time indices are not the same for all signals.
In this format we expect the following columns to be present:

`TIME, ID, TYPE, VALUE`

Those columns stand for:

* TIME is a datetime column
* ID specificies the object (Usually a location)
* TYPE what is measured (Precipitation, flow or vibrations)
* VALUE the measured sensor value

For example:

+----------------------+--------+-------+--------+
| TIME                 | ID     | TYPE  | VALUE  |
+======================+========+=======+========+
| 2020-01-01 00:00:00  | Pump1  | Flow  | 2      |
+----------------------+--------+-------+--------+
| 2020-01-01 00:02:00  | Pump1  | Speed | 800    |
+----------------------+--------+-------+--------+
| ...                  | ...    | ...   | ...    |
+----------------------+--------+-------+--------+

Wide format
-----------

This format is used by the models and feature engineers. It usually has the following columns:

`TIME, ID1_TYPE1, ID1_TYPE2, ID_TYPE1, ...`

Note that `_` is the default separator, but it can be changed in the reshaping functions, in case your ID or TYPE contains an underscore.

The columns stand for:

* TIME is a datetime column
* ID_TYPE Columns signify the different sensor values per ID and TYPE

For example:

+----------------------+-------------+--------------+-------------+------+
| TIME                 | Pump1_Flow  | Pump1_Speed  | Pump2_Flow  | ...  |
+======================+=============+==============+=============+======+
| 2020-01-01 00:00:00  | 2           | 800          | 0           | ...  |
+----------------------+-------------+--------------+-------------+------+
| 2020-01-01 00:02:00  | 3           | 802          | 0           | ...  |
+----------------------+-------------+--------------+-------------+------+
| ...                  | ...         | ...          | ...         | ...  |
+----------------------+-------------+--------------+-------------+------+


Timezone
--------

SAM does not need timezone information to work. However some feature engineering functions (like using datetime components) do expect the data to be in UTC (with or without tz info).
Therefore it is *strongly*  encouraged to make sure your data in in UTC, to prevent summer/winter-time issues. The feature engineer can take the local time into account while the data remains in UTC.
