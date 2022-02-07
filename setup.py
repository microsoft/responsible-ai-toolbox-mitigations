from setuptools import setup, find_packages

setup(
    name="raimitigations",
    version="0.0.0",
    author="Akshara Ramakrishnan",
    author_email="aksharar@microsoft.com",
    description="Data Balance Analysis and Error mitigation steps on python",
    packages=find_packages(
        include=[
            "raimitigations.databalanceanalysis",
            "raimitigations.dataprocessing",
            "test",
        ]
    ),
    install_requires=[
        "pandas",
        "numpy",
        "wheel",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
    ],
)
