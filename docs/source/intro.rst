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

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation.
To be sure, run the command again to install the latest version.

Configuration
-------------

A configuration file can be created as ``.config``. This configuration file only stores api credentials for weather api's for now.
The configuration file should be placed in your working directory, and  is parsed using the
`Python3 configparser <https://docs.python.org/3/library/configparser.html>`_, and an example configuration is shown below:

.. code-block:: none

    [regenradar]
    user=loes.knoben
    password=secret123

    [openweathermap]
    apikey=secret456

Main components
---------------
SAM aims to support the whole analysis from ingesting data to measuring and visualising model performance.
The package is build up accordingly as can be seen in the Contents below.

Some highlights:

* Easily train a functional model on timeseries data with ``sam.models.SamQuantileMLP``
* Build features from time series using the ``sam.feature_engineering.BuildRollingFeatures`` class
* Automatically remove extreme values, flatlines and missing values with ``sam.validation.create_validation_pipe``
* Do quantile regression in Keras, with ``sam.metrics.keras_joint_mse_tilted_loss``
* Visualise the flags raised by the model in a ``sam.visualization.plot_incident_heatmap``

Contributing / requesting features
----------------------------------
Contributing works by forking  `the repository <https://github.com/RoyalHaskoningDHV/sam/fork>`_ and creating a pull request if you want to incorporate your changes into the code.
Also see the `CONTRIBUTING.md <https://github.com/RoyalHaskoningDHV/sam/blob/main/CONTRIBUTING.md>`_
file in the repository for more information. 

We keep track of new features, bug reports and progress on `the GitHub issues page <https://github.com/RoyalHaskoningDHV/sam/issues>`_.
