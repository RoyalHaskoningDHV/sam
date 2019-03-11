from setuptools import setup, find_packages

setup(name='sam',
      version='0.3.0',
      description='Ynformed package for smart asset management',
      long_description=open('README.md').read(),
      url='https://dev.ynformed.nl/diffusion/78/',
      author='Ynformed',
      author_email='fenno@ynformed.nl',
      license='',
      packages=find_packages(exclude=['*tests']),
      zip_safe=True,
      install_requires=['pandas', 'numpy'],
      extras_require={
          'feature_engineering': ['scipy'],
          'metrics': ['sklearn'],
          'utils': ['pymongo'],
          'visualization': ['matplotlib', 'seaborn', 'sklearn'],
      },
      tests_require=['pytest', 'pytest-cov', 'pytest-mpl']
      )
