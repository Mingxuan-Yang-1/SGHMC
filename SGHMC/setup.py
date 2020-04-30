import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="sghmc_pkg_663_2.0.0",
    version="2.0.0",
    author="Jiawei Chen & Mingxuan Yang",
    author_email="",
    description="A package for HMC, SGHMC, SGLD and SGD with momentum.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mingxuan-Yang/SGHMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)