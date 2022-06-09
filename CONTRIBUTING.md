# Contributing

This project is a community effort, and everyone is welcome to contribute. The package is moderated by Royal HaskoningDHV. Before contributing, please read this guide.

## Links

- [GitHub repository](https://github.com/RoyalHaskoningDHV/sam)
- [Documentation](https://samdocs.digitalapps.royalhaskoningdhv.com)
- [Issues and feature requests](https://github.com/RoyalHaskoningDHV/sam/issues)
- [Pull requests](https://github.com/RoyalHaskoningDHV/sam/pulls)
- [Community (discussions, Q&A))](https://github.com/RoyalHaskoningDHV/sam/discussions)


## Bug reports and feature requests

The single most important contribution that you can make is to report bugs and make feature requests. The development work on sam is largely driven by these, so please make your voice heard! Any bugs/feature requests [can be created here.](https://github.com/RoyalHaskoningDHV/sam/issues) No permission is needed to create an issue. Please use user stories for feature requests. Be as clear as you can in the description. 

In the case of bug reports please fill in a small script to reproduce the problem in the bug report. Report your sam version, python version, and the operating system you are using as well.

## Developing.

We recommend to use a virtual environment for a clean slate and to avoid any conflicts with other packages. For example, to create a virtual environment, run `python3 -m venv <env name>`. Then, run `source <env name>/bin/activate`.

If you want to develop this package, you wil need to install a local version using pip. This is done by cloning the repo, going to the root folder of this package, and running `pip install -e .` This will install a development version of the package locally. That means that if you make local changes, the package will automatically reflect them. 

If you want to develop in a Jupyter notebook, you will also need to reload the sam package whenever you run `from sam import x`. This can be achieved by putting the following lines at the top of every notebook:

```
lang=python
%load_ext autoreload
%autoreload 2
```

This will reload sam every time you run a new cell. For more information abut the autoreload extension, [see the documentation here](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html)

## Linting

Any Pull Request must pass linting requirements. To test this locally, first install the dependencies

```
lang=bash
pip install flake8
```

Then, to run the linter manually, go to the root folder of the project, and run `flake8 sam --config=setup.cfg`. Satisfying all flake8 rules is required when developing this package.

## Testing

Any Pull Request must pass unit tests. To test this locally, first install the dependencies

```
lang=bash
pip install pytest
pip install pytest-cov
pip install pytest-mpl
pip install fastparquet
```

Then run the unit tests using `pytest`.

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
To ensure quality of code and documentation, we utilise standard definitions of done for SAM. You are kindly requested to comply with these when making a Pull Request.

### Code implementations

* Functionality implemented in code
* Code actually tested by using it at least once
* Appropriate unit tests written
* Code documented including example use
* Code style follows linter rules
* Updated `CHANGELOG.md` and version number in `setup.cfg`
* Code reviewed and approved

If functionality supports a new step in the analyses:

* Update general documentation to incorporate a reference to this code

### When specifically writing documentation

* Documentation written
* Documentation build tested locally
* Reviewed and approved
