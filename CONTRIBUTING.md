# Contributing

sam is an internal package being developed by Ynformed. For any new code, code review by another Ynformed is therefore required!

## Bug Reports and Feature Requests

The single most important contribution that you can make is to report bugs and make feature requests. The development work on sam is largely driven by these, so please make your voice heard! Any bugs/feature requests [can be created here.](https://dev.azure.com/ynformed/SAM/_backlogs/backlog/SAM%20Team/Epics) No permission is needed to create a card, so go nuts! I kindly ask you prefix your cards with `[bug]` or `[feature]`, and alert one of the developers since we don't check the board every day!

Bug reports are most helpful if you send us a script which reproduces the problem.

## Developing

If you want to develop this package, you wil need to install a local version using pip. This is done by cloning the repo, going to the root folder of this package, and running `pip install -e .` This will install a development version of the package locally. That means that if you make local changes, the package will automatically reflect them. 

If you want to develop in a Jupyter notebook, you will also need to reload the sam package whenever you run `from sam import x`. This can be achieved by putting the following lines at the top of every notebook:

```
lang=python
%load_ext autoreload
%autoreload 2
```

This will reload sam everytime you run a new cell. For more information abut the autoreload extension, [see the documentation here](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html)

## Linting

Linting is done automatically by devops during a PR. To do this, first install the dependencies

```
lang=bash
pip install pycodestyle
```

Then, to run the linter manually, go to the root folder of the project, and run `pycodestyle`. Alternatively, you can use flake8, but satisfying all flake8 rules is not required when developing this package.

## Testing

Unit tests are ran automatically by devops during a PR. To do this, first install the dependencies

```
lang=bash
pip install pytest
pip install pytest-cov
pip install pytest-mpl
```

## Documentation

This documentation is built automatically after every commit by Azure Devops, with no interaction required. If you want to build it yourself locally, first install the dependencies:

```
lang=bash
pip install sphinx
pip install sphinx_rtd_theme
pip install numpydoc
pip install recommonmark
pip install sphinx-markdown-tables
```

Then, run the command: `sphinx-build -b html docs/source/ docs/build/`

## Definitions of done
To ensure quality of code and documentation, we utilise standard definitions of done for SAM. You are kindly requested to comply with these when making a diff.

### Code implementations

* Functionality implemented in code
* Code actually tested by using it at least once
* Code documented including example use
* Appropriate unit tests written
* Code reviewed and landed

If functionality supports a new step in the analyses:

* Update general documentation to incorporate a reference to this code

### When specifically writing documentation

* Documentation written
* Build tested locally
* Reviewed and landed
