#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="thesis-2048-rl",
    version="0.1.0",
    description="Reinforcement Learning for 2048 Game",
    author="Davide",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "imageio>=2.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 