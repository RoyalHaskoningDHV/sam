The SAM Python package is a collection of tools and functions mainly for **sensor data or time series analysis**, and by extension for **smart asset management** analysis. All functionality has been tested and documented. Using this ensures a company wide generic approach, and will greatly speed up analysis.

Getting started
---------------

To install the package, you need to have access to phabricator via git. Then, you can install it with the following command:

```python
pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git
# To install all optional dependencies: (such as pymongo, sklearn, seaborn)
pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git#egg=sam[all]
```

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the command again to install the latest version. Many other resources related to sam can [be found on the landing page here.](https://main-sam.ynformed.nl/)

Main components
---------------
SAM aims to support the whole analysis from ingesting data to measuring and visualising model performance. The package is build up accordingly as can be seen in the Contents below. If possible, we follow the `Scikit-learn <https://scikit-learn.org>`_ API making components compatible with, for example, Pipelines.

Some highlights:

* Explore correlations in the data using functions in ``sam.feature_selection.top_correlation``
* Build features from time series using the ``sam.feature_engineering.BuildRollingFeatures`` class
* Score how many incidents a model is able to find using ``sam.metrics.incident_recall``
* Visualise the flags raised by the model in a ``sam.visualization.make_incident_heatmap``
* ...

Example use cases
-----------------

Nereda incident prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^
In the Nereda water treatment plant, sensors measure critical output values. These values should not exceed certain thresholds, as the client might then be fined. We use SAM to build a predictive model, warning several batches in advance for a high risk of exceeding threshold.

In this case we do everything but the actual modelling with SAM functionality. The modelling is done in Scikit-learn.

Getting started
---------------
If you want to use or develop this package, you will need to install a local version using pip.
This is done by cloning the repo, going to the root folder of this package and running ``pip install -e .``
This will install a development version of the package locally. That means that if you
make local changes, the package will automatically reflect them. 

If you then want to use the package in Python, use ``from sam import x``.

Contributing / requesting features
----------------------------------
Contributing works by cloning  `the repository <https://dev.ynformed.nl/diffusion/78/>`_ and using 
`arcanist` for pushing to the repo. See the repository README for more information. 

We keep track of new features and progress on https://dev.ynformed.nl/tag/sam_platform/.
Features may be requested by contacting Tim (preferably via Slack).