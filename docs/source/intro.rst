The SAM Python package is a collection of tools and functions mainly for **sensor data or time series analysis**,
and by extension for **smart asset management** analysis. All functionality has been tested and documented.
Using this ensures a generic approach, and will greatly speed up analysis.

Getting started
---------------

The easiest way to install the package is using pip:

.. code-block:: bash

	pip install sam
	# To install all optional dependencies: (such as pymongo, seaborn, tensorflow, etc.)
	pip install sam[all]

There are different optional dependencies for SAM, if you are unsure use `pip install 'sam[all]'` other options include `plotting` (just use the plotting functionality), `data_science` (all dependencies needed for a data scientist) and `data_engineering` (dependencies for data engineer).

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the `pip install -U sam` command to install the latest version.

Simple example
--------------

Below you can find a simple example on how to use one of our timeseries models. For more examples, check our `example notebooks <https://github.com/RoyalHaskoningDHV/sam/tree/main/examples>`_

.. code-block:: python

    from sam.datasets import load_rainbow_beach
    from sam.models import MLPTimeseriesRegressor
    from sam.feature_engineering import SimpleFeatureEngineer

    data = load_rainbow_beach()
    X, y = data, data["water_temperature"]

    # Easily create rolling and time features to be used by the model
    simple_features = SimpleFeatureEngineer(
        rolling_features=[
            ("wave_height", "mean", 24),
            ("wave_height", "mean", 12),
        ],
        time_features=[
            ("hour_of_day", "cyclical"),
        ],
        keep_original=False,
    )

    # Define your model, see the docs for all parameters
    model = MLPTimeseriesRegressor(
        predict_ahead=(1, 2, 3), # Multiple predict aheads are possible
        quantiles=(0.025, 0.975), # Predict quantile bounds for anomaly detection
        feature_engineer=simple_features,
        epochs=20,
    )
    model.fit(X, y)


Configuration
-------------

A configuration file can be created as ``.config``. This configuration file only stores api credentials for weather api's for now.
The configuration file should be placed in your working directory, and  is parsed using the
`Python3 configparser <https://docs.python.org/3/library/configparser.html>`_, and an example configuration is shown below:

.. code-block:: ini

    [regenradar]
    url=https://rhdhv.lizard.net/api/v3/raster-aggregates/?
    user=user.name
    password=secret

    [openweathermap]
    apikey=secret

Main components
---------------
SAM aims to support the whole analysis from ingesting data to measuring and visualising model performance.
The package is build up accordingly as can be seen in the Contents below.

Some highlights:

* Easily train a functional model on timeseries data with ``sam.models.MLPTimeseriesRegressor``
* Build features from time series using the ``sam.feature_engineering.SimpleFeatureEngineer`` class
* Automatically remove extreme values, flatlines and missing values with ``sam.validation.create_validation_pipe``
* Visualise the predicted quantiles using ``sam.visualization.sam_quantile_plot``

Contributing / requesting features
----------------------------------
Contributing works by forking  `the repository <https://github.com/RoyalHaskoningDHV/sam/fork>`_ and creating a pull request if you want to incorporate your changes into the code.
Also see the `CONTRIBUTING.md <https://github.com/RoyalHaskoningDHV/sam/blob/main/CONTRIBUTING.md>`_
file in the repository for more information. 

We keep track of new features, bug reports and progress on `the GitHub issues page <https://github.com/RoyalHaskoningDHV/sam/issues>`_.
