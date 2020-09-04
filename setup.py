#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
from setuptools.command.test import test as TestCommand
import sys


# versioning
MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = False
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"


# are we on windows, darwin, etc?
platform = sys.platform
packages = find_packages()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = 'refnx'

    def run_tests(self):
        import shlex
        import pytest
        print("Running tests with pytest")
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


with open("README.md", "r") as fh:
    long_description = fh.read()


# refnx setup
info = {
        'name': 'minimize',
        'description': '',
        'long_description': '',
        'long_description_content_type': "text/markdown",
        'author': 'Andrew Nelson',
        'author_email': 'andyfaff+refnx@gmail.com',
        'license': 'BSD',
        'url': 'https://github.com/andyfaff/Minimize',
        'project_urls': {"Bug Tracker": "https://github.com/andyfaff/Minimize/issues",
                         "Source Code": "https://github.com/andyfaff/Minimize"},
        'classifiers': [],
        'packages': packages,
        'include_package_data': True,
        'python_requires': '>=3.7',
        'cmdclass': {'test': PyTest},
        'entry_points': {"gui_scripts" : ['refnx = refnx.reflect:main',
                                          'slim = refnx.reduce:main']}
        }

####################################################################
# this is where setup starts
####################################################################
def setup_package():
    setup(**info)


if __name__ == '__main__':
    setup_package()
