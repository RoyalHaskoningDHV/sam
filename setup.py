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
    author="Royal HaskoningDHV",
    author_email="arjan.bontsema@rhdhv.com",
    license="BSD-3-Clause",
    description="Time series anomaly detection and forecasting",
    long_description=open("README.md").read(),
    url="https://github.com/RoyalHaskoningDHV/sam",
    project_urls={
        "Q&A": "https://github.com/RoyalHaskoningDHV/sam/discussions",
        "Issues": "https://github.com/RoyalHaskoningDHV/sam/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["*tests"]),
    python_requires=">=3.6",
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
            "Jinja2==3.0.3",
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
            "Jinja2==3.0.3",
            "statsmodels",
        ],
    },
    tests_require=["pytest", "pytest-cov", "pytest-mpl", "fastparquet"],
)
