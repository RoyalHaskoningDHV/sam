import configparser

from setuptools import find_packages, setup

config = configparser.ConfigParser()
config.read("setup.cfg")
version = (
    config["metadata"]["major-version"] + "." + config["metadata"]["minor-version"]
)

setup(
    name="sam",
    version=version,
    description="Time series anomaly detection and forecasting",
    long_description=open("README.md").read(),
    url="https://dev.azure.com/corporateroot/SAM",
    author="Royal HaskoningDHV",
    author_email="arjan.bontsema@rhdhv.com",
    license="",
    packages=find_packages(exclude=["*tests"]),
    zip_safe=True,
    install_requires=["pandas>=1.1.0", "numpy>=1.13", "scikit-learn<0.24.0"],
    extras_require={
        "all": [
            "matplotlib",
            "cloudpickle",
            "nfft",
            "pymongo",
            "requests",
            "scipy",
            "seaborn",
            "tensorflow<=2.3.1",
            "eli5",
            "shap",
            "plotly",
            "statsmodels",
        ],
        "plotting": ["matplotlib", "plotly", "seaborn"],
        "data_engineering": ["requests", "pymongo"],
        "data_science": [
            "tensorflow<=2.3.1",
            "cloudpickle",
            "nfft",
            "scipy",
            "shap",
            "eli5",
            "statsmodels",
        ],
    },
    tests_require=["pytest", "pytest-cov", "pytest-mpl", "fastparquet"],
)
