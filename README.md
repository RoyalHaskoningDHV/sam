# SAM

SAM is a Python package for timeseries analysis, anomaly detection and forecasting.

Author: [Royal HaskoningDHV](https://global.royalhaskoningdhv.com/digital)

Email: [ruben.peters@rhdhv.com](mailto:ruben.peters@rhdhv.com)

## Getting started

The documentation is available [here.](https://sam-rhdhv.readthedocs.io/en/latest/).

The easiest way to install is package is using pip:
```
pip install sam
```

There are different optional dependencies for SAM, if you are unsure use `pip install sam[all]` other options include `plotting` (just use the plotting functionality), `data_science` (all dependencies needed for a data scientist) and `data_engineering` (dependencies for data engineer).

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the `pip install -U sam` command to install the latest version.

## Configuration

A configuration file can be created as `.config`. This configuration file only stores api credentials for now, but more options may be added in the future. The configuration file is parsed using the [Python3 configparser](https://docs.python.org/3/library/configparser.html), and an example configuration is shown below:

```
[regenradar]
user=regenradar.username
password=secret123

[openweathermap]
apikey=secret456
```

## Issue tracking and Feature Requests

Anyone can create feature requests or bug reports! You can browse and create new issues on GitHub: https://github.com/RoyalHaskoningDHV/sam/issues
