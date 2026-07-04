from setuptools import setup, find_packages

setup(
    name="tacimag",
    version="0.1.0",
    description="TacImag: Imagining the Sense of Touch for Robotic Manipulation",
    packages=find_packages(include=["tacimag", "tacimag.*"]) + ["hydra_plugins"],
    package_data={"tacimag": ["config/**/*.yaml"]},
    python_requires=">=3.8",
    # only the direct imports not already guaranteed by the ManiFeel install
    install_requires=[
        "threadpoolctl",
        "filelock",
    ],
)
