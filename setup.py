# -*- coding: utf-8 -*-
__author__ = 'John Bangsund'
import os
from setuptools import setup, find_packages

description = ("Package for common calculations for organic light-emitting"
              "devices, such as color coordinates, transfer matrix modeling "
              "of electric field profiles within a device, and light outcoupling")

setup(
    name="oledpy",
    version='0.0.1',
    author="John Bangsund",
    author_email="jsb@umn.edu",
    description=description,
    license="",
    keywords="oleds, outcoupling, optics, fresnel, transfer matrix method",
    url="",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"],
    install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'pandas'
      ],
    zip_safe=False
)
