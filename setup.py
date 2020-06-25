from setuptools import setup, Extension
import setuptools
import os
import sys

# get __version__, __author__, and __email__
exec(open("./epipack/metadata.py").read())

setup(
    name='epipack',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/benmaier/epipack',
    license=__license__,
    description="Build any polynomial epidemiological model, analyze its ODEs, or run stochastic simulations on networks or well-mixed systems.",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
                'numpy>=1.17',
                'scipy>=1.3',
                'sympy==1.6',
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'
                 ],
    project_urls={
        'Documentation': 'TODO',
        'Contributing Statement': 'https://github.com/benmaier/epipack/blob/master/CONTRIBUTING.md',
        'Bug Reports': 'https://github.com/benmaier/epipack/issues',
        'Source': 'https://github.com/benmaier/epipack/',
        'PyPI': 'https://pypi.org/project/epipack/',
    },
    include_package_data=True,
    zip_safe=False,
)
