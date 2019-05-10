The SAM Python package is a collection of tools and functions mainly for **sensor data or time series analysis**, and by extension for **smart asset management** analysis. All functionality has been tested and documented. Using this ensures a company wide generic approach, and will greatly speed up analysis.

Getting started
---------------

To install the package, you need to have access to phabricator via git. Then, you can install it with the following command:

.. code-block:: bash

	pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git
	# To install all optional dependencies: (such as pymongo, knmy, seaborn)
	pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git#egg=sam[all]

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the command again to install the latest version. Many other resources related to sam can be found on the `landing page here. <https://main-sam.ynformed.nl/>`_

Configuration
-------------

A configuration file can be created as ``.config``. This configuration file only stores api credentials for now, but more options may be added in the future. The configuration file is parsed using the `Python3 configparser <https://docs.python.org/3/library/configparser.html>`_, and an example configuration is shown below:

.. code-block:: none

    [regenradar]
    user=loes.knoben
    password=secret123

    [openweathermap]
    apikey=secret456

Main components
---------------
SAM aims to support the whole analysis from ingesting data to measuring and visualising model performance. The package is build up accordingly as can be seen in the Contents below.

Some highlights:

* Explore correlations in the data using functions in ``sam.exploration.top_correlation``
* Build features from time series using the ``sam.feature_engineering.BuildRollingFeatures`` class
* Score how many incidents a model is able to find using ``sam.metrics.incident_recall``
* Visualise the flags raised by the model in a ``sam.visualization.plot_incident_heatmap``
* ...

Contributing / requesting features
----------------------------------
Contributing works by cloning  `the repository <https://dev.ynformed.nl/diffusion/78/>`_ and using 
`arcanist` for pushing to the repo. See the ``CONTRIBUTING.md`` file in the repository for more information. 

We keep track of new features and progress on https://dev.ynformed.nl/tag/sam_platform/.
Features may be requested by contacting Tim or Fenno (preferably via Slack).
