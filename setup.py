from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

import re
VERSIONFILE="payequity/version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='PayEquity',
    version=verstr,
    author='Steven Cunden',
    author_email='slcunden@gmail.com',
    description='Pay Equity Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/CW-People-Analytics/CWPayEquity',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy','pandas'],
)
