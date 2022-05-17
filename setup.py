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
    python_requires=">=3.8",
    install_requires=["pandas~=1.3", "numpy~=1.21", "scikit-learn~=1.0"],
    extras_require={
        "all": [
            "matplotlib",
            "cloudpickle",
            "nfft",
            "pymongo",
            "requests",
            "scipy",
            "seaborn",
            "tensorflow~=2.8.0",
            "eli5",
            "Jinja2~=3.0.3",
            "shap",
            "plotly",
            "statsmodels",
        ],
        "plotting": ["matplotlib", "plotly", "seaborn"],
        "data_engineering": ["requests", "pymongo"],
        "data_science": [
            "tensorflow~=2.8.0",
            "cloudpickle",
            "nfft",
            "scipy",
            "shap",
            "eli5",
            "Jinja2~=3.0.3",
            "statsmodels",
        ],
    },
    tests_require=["pytest", "pytest-cov", "pytest-mpl", "fastparquet"],
)
