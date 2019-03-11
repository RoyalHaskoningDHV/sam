## Version: 0.3.0

# sam description

The Ynformed library for smart asset management.

Author: Fenno Vermeij

Email: fenno@ynformed.nl

Contributors: Fenno, Loes, Rutger, Sebastiaan, Tim

# Getting started

Documentation is [available here.](10.2.0.20/sam) This is on the ynformed internal server. This server is only reachable from the ynformed office wifi, or via vpn. Many other resources related to sam can [be found on the landing page here.](https://main-sam.ynformed.nl/)

To install the package, you need to have access to phabricator via git. Then, you can install it with the following command:

```
pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git
```

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the command again to install the latest version.

# Developing

If you want to develop this package, you wil need to install a local version using pip. This is done by going to the root folder of this package, and running `pip install -e .` This will install a development version of the package locally. That means that if you make local changes, the package will automatically reflect them. 

If you want to develop in a Jupyter notebook, you will also need to reload the sam package whenever you run `from sam import x`. This can be achieved by putting the following lines at the top of every notebook:

```
lang=python
%load_ext autoreload
%autoreload 2
```

This will reload sam everytime you run a new cell. For more information abut the autoreload extension, [see the documentation here](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html)

## Linting

Linting is done automatically by arcanist before a diff. To do this, first install the dependencies

```
lang=bash
pip install pycodestyle
```

Then, to run the linter manually, go to the root folder of the project, and run `pycodestyle`. Alternatively, you can use flake8, but satisfying all flake8 rules is not required when developing this package.

## Testing

Unit tests are ran automatically by arcanist before a diff. To do this, first install the dependencies

```
lang=bash
pip install pytest
pip install pytest-cov
pip install pytest-mpl
```

Additionally, you will have to install the PytestMPLTestEngine extension. Download the file from `https://dev.ynformed.nl/P8`, and copy it to `%ARCANIST_DIR%/src/extensions/PytestMPLTestEngine.php`. Then, to run the tests manually, go the the root folder of the project, and run `arc unit`. If any of the visualizations were changed, the baseline images have to be rebuilt. This can be done with: `pytest --mpl-generate-path=sam/visualization/tests/baseline`.

## Documentation

This documentation is built automatically after every commit [by jenkins here](10.2.0.20/sam), with no interaction required. If you want to build it yourself locally, first install the dependencies:

```
lang=bash
pip install sphinx
pip install sphinx_rtd_theme
pip install numpydoc
pip install recommonmark
```

Then, go to the /docs folder, and run the command: `sphinx-build -b html source/ build/`
