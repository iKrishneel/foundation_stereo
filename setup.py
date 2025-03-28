#! /usr/bin/env python

from setuptools import find_packages, setup

try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


install_requires = []


__name__ = 'foundation_stereo'
__version__ = '0.0.1'

setup(
    name=__name__,
    url='https://github.com/iKrishneel/foundation_stereo',
    version=f'{__version__}',  # NOQA: F821
    long_description=readme,
    packages=find_packages(),
    long_description_content_type="text/markdown",
    zip_safe=True,
    install_requires=install_requires,
    test_suite='tests',
    include_package_data=True,
    # package_data={__name__: [f'{__name__}/configs/config.yaml']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
