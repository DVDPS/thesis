#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="thesis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
        # Add other dependencies as needed
    ],
    python_requires=">=3.6",
    description="Reinforcement Learning for 2048 Game",
    author="Davide",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 