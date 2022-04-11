from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="raimitigations",
    version="0.0.3",
    author="Akshara Ramakrishnan",
    author_email="aksharar@microsoft.com",
    description="Data Balance Analysis and Error mitigation steps on python",
    packages=find_packages(
        exclude=["test"],
        include=["raimitigations.databalanceanalysis", "raimitigations.dataprocessing"],
    ),
    install_requires=[
        "pandas",
        "numpy",
        "wheel",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
