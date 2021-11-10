# Copyright (c) Microsoft Corporation and Fairlearn contributors.

import setuptools
import errorsmitigation

# Fetch ReadMe
with open("README.md", "r") as fh:
    long_description = fh.read()

# Use requirements.txt to set the install_requires
with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f]

setuptools.setup(
    name=errorsmitigation.__name__,
    version=errorsmitigation.__version__,
    author="Dany Rouhana", # todo: add more team members
    author_email="danyr@microsoft.com", # todo: change this email to a project email
    description="Algorithms for mitigating unfairness in supervised machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TBD", # todo: git hub url
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Microsoft License",
        "Operating System :: OS Independent",
        "Development Status :: Alpha"
    ],
    include_package_data=True,
    zip_safe=False,
)
