from setuptools import setup, find_packages

setup(
    name="watchdog_ai",
    version="0.1.0",
    packages=find_packages(where="src", exclude=["tests*"]),
    package_dir={"": "src"}
)