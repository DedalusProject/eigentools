import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eigentools", 
    version="2.2112",
    author="J. S. Oishi",
    author_email="jsoishi@gmail.com",
    description="A toolkit for solving eigenvalue problems with Dedalus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dedalusproject/eigentools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    python_requires='>=3.5',
)
