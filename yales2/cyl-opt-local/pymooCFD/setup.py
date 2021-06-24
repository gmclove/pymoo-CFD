import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="pymooCFD-gmclove",
    version="0.0.1",
    author="George Love",
    author_email="george.love077@gmail.com",
    description="Interface between pymoo and CFD solver.",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gmclove/pymoo-CFD",
    project_urls={
        "Bug Tracker": "https://github.com/gmclove/pymoo-CFD/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License :: Version 2.0, January 2004",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
