# sam

The Ynformed library for smart asset management.

Author: Fenno Vermeij

Email: fenno@ynformed.nl

Contributors: Fenno, Daan, Rutger, Arjan, Loes, Tim, Sebastiaan

## Getting started

Documentation is [available here.](http://10.2.0.20/sam) This is on the ynformed internal server. This server is only reachable from the ynformed office wifi, or via vpn.

To install the package, you need to have access to phabricator via git. Then, you can install it with the following command:

```
lang=python
pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git
# To install all optional dependencies: (such as pymongo, knmy, seaborn)
pip install git+ssh://git@dev.ynformed.nl:2222/diffusion/78/sam.git#egg=sam[all]
```

Keep in mind that the sam package is updated frequently, and after a while, your local version may be out of date with the online documentation. To be sure, run the command again to install the latest version. Many other resources related to sam can [be found on the landing page here.](https://main-sam.ynformed.nl/)

## Configuration

A configuration file can be created as `.config`. This configuration file only stores api credentials for now, but more options may be added in the future. The configuration file is parsed using the [Python3 configparser](https://docs.python.org/3/library/configparser.html), and an example configuration is shown below:

```
[regenradar]
user=loes.knoben
password=secret123

[openweathermap]
apikey=secret456
```

## Issue tracking and Feature Requests

Anyone can create feature requests or bug reports! You can browse and create new issues in Phabricator: https://dev.ynformed.nl/project/view/22/
