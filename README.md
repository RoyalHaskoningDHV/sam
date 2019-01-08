## Version: 0.2.0

# sam description

The Ynformed library for smart asset management.

Author: Sebastiaan Grasdijk
Email: sebastiaan@ynformed.nl

Reviewers: Fenno, Rutger, Tim

# Run options

``` 
```

# Linting

Linting is done automatically by arcanist before a diff. To do this, first install
the dependencies

```
pip install pep8
```

Then, to run the linter manually, go to the root folder of the project, and run `pep8`.

# Testing

Unit tests are ran automatically by arcanist before a diff. To do this, first install
the dependencies

```
pip install pytest
pip install pytest-cov
```

Then, to run the tests manually, go the the root folder of the project, and run `pytest`.

# Documentation

Documentation is available on 10.2.0.20/sam . This is on the ynformed internal server.
This server is only reachable from the ynformed office wifi, or via vpn.
This documentation is built automatically after every commit, with no interaction required.
If you want to build it yourself locally, first install the dependencies:

```
 pip install sphinx
 pip install sphinx_rtd_theme
 pip install numpydoc
 pip install recommonmark
```

Then, go to the /docs folder, and run the command: `sphinx-build -b html source/ build/`

# Developing

If you want to develop this package, you wil need to install a local version using pip.
This is done by going to the root folder of this package, and running `pip install -e .`
This will install a development version of the package locally. That means that if you
make local changes, the package will automatically reflect them. 

If you want to develop in a notebook, you will also need to reload the sam package 
whenever you run `from sam import x`. This can be achieved by putting the following
lines at the top of every notebook:
```
%load_ext autoreload
%autoreload 2
%aimport sam
```
This will reload sam everytime you run a new cell. The third line is optional: if you 
leave it out, you will reload every import every cell, instead of only those from sam.

