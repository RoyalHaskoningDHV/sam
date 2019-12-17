The SAM Python package is a collection of tools and functions mainly for **sensor data or time series analysis**,
and by extension for **smart asset management** analysis. All functionality has been tested and documented.
Using this ensures a company wide generic approach, and will greatly speed up analysis.

Getting started
---------------

To install the package, you need to have access to phabricator via git. Then, you can install it with the following command:

.. code-block:: bash

	pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git
	# To install all optional dependencies: (such as pymongo, knmy, seaborn)
	pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git#egg=sam[all]

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
* ...

Contributing / requesting features
----------------------------------
Contributing works by cloning  `the repository <https://dev.ynformed.nl/diffusion/78/>`_ and using 
`arcanist` for pushing to the repo. See the `CONTRIBUTING.md <https://dev.ynformed.nl/diffusion/78/browse/master/CONTRIBUTING.md>`_
file in the repository for more information. 

We keep track of new features and progress on `the phabricator board <https://dev.ynformed.nl/tag/sam_platform/>`_.
Features may be requested by contacting Martijn, or any of the SAM developers (preferably via Slack).
