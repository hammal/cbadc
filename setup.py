import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cbc",  # Replace with your own username
    version="0.0.1",
    author="Hampus Malmberg",
    author_email="hampus.malmberg88@gmail.com",
    description="A toolbox for simulating control-bounded converters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hammal/cbc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
