from setuptools import find_packages, setup

setup(
    name="simpler_env",
    version="0.0.1",
    author="Xuanlin Li",
    packages=find_packages(include=["simpler_env*, third_party/rdt*"]),
    python_requires=">=3.10",
)
