# -*- coding: utf-8 -*-
import os
from setuptools import setup

# Get README
with open("README.rst") as f:
    readme = f.read()

setup(
    name="resmda",
    description="A simple 2D reservoir modeller plus ESMDA.",
    long_description=readme,
    author="Dieter Werthmüller, Gabriel Serrao Seabra",
    author_email="info@emsig.xyz",
    url="https://github.com/tuda-geo/resmda",
    license="Apache-2.0",
    packages=["resmda"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
    ],
    python_requires=">=3.10",
    install_requires=[
        "scipy",
        "scooby",
    ],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("resmda", "version.py"),
    },
    setup_requires=["setuptools_scm"],
)
