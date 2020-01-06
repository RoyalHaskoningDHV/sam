from setuptools import setup, find_packages
import configparser
import os

config = configparser.ConfigParser()
config.read('setup.cfg')
version = config['metadata']['major-version'] + '.' + config['metadata']['minor-version']

setup(name='sam',
      version=version,
      description='Ynformed package for smart asset management',
      long_description=open('README.md').read(),
      url='https://dev.ynformed.nl/diffusion/78/',
      author='Ynformed',
      author_email='fenno@ynformed.nl',
      license='',
      packages=find_packages(exclude=['*tests']),
      zip_safe=True,
      install_requires=['pandas>=0.23', 'numpy>=1.13', 'scikit-learn>=0.20'],
      extras_require={
          'all': ['knmy', 'matplotlib', 'nfft', 'pymongo', 'requests', 'scipy', 'seaborn',
                  'tensorflow']
      },
      tests_require=['pytest', 'pytest-cov', 'pytest-mpl']
      )
