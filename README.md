## Version: 0.1.0

# sam description

The Ynformed library for smart asset management.

Author: Sebastiaan Grasdijk
Email: sebastiaan@ynformed.nl

Reviewers: Fenno, Rutger, Tim

# Run options

``` 
```

# Testing

```
```

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


# Before committing
When committing, first rebuild the documentation. This can be done through the following command within the /docs folder:


```
sphinx-build -b html source/ build/
```

You will need to have the following packages:
```
 pip install sphinx
 pip install sphinx_rtd_theme
 pip install recommonmark
```


