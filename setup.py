"""Packaging settings."""


from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from gkdb import __version__


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=gkdb', '--cov-report=term-missing',
                      '--ignore=lib/'])
        raise SystemExit(errno)


setup(
    name = 'gkdb',
    version = __version__,
    description = 'Peewee-based tools for interacting with the GyroKinetic DataBase (GKDB).',
    long_description = long_description,
    url = 'https://github.com/gkdb/gkdb',
    author = 'Karel van de Plassche',
    author_email = 'karelvandeplassche@gmail.com',
    license = 'MIT',
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: MIT',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
    ],
    packages = find_packages(exclude=['docs', 'tests*']),
    install_requires = ['peewee', 'numpy', 'scipy', 'IPython', 'psycopg2', 'pandas', 'xarray'],
    extras_require = {
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    cmdclass = {'test': RunTests},
)
