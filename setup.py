import os
from sys import path
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

# this grabs the requirements from requirements.txt
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

# scan the 'cbc' directory for extension files, converting
# them to extension names in dotted notation


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        '.'.join(extName.split('.')[1:]),
        [extPath],
        # adding the '.' to include_dirs is CRUCIAL!!
        include_dirs=[".", "src/cbc/analog_system",
                      "src/cbc/digital_control", "src/cbc/analog_signal", "src/cbc/digital_estimator"],
        extra_compile_args=["-fopenmp"],
        # extra_link_args=['-g'],
        libraries=["m"],
    )


# get the list of extensions
extNames = scandir("src/cbc")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

# extensions += [Extension(
#     'cbc.parallel_digital_estimator',
#     sources=['src/cbc/parallel_digital_estimator/parallel_digital_estimator.pyx'],
#     include_dirs=['src/cbc/parallel_digital_estimator'],
#     language='c++',
#     extra_compile_args=['-fopenmp'])]

print(extensions)

# extensions = [
#     Extension(
#         "cbc.circuit_simulator",
#         ["src/cbc/{analog_system,analog_signal,digital_control,circuit_simulator}.pyx"],
#         # include_dirs=[numpy.get_include(), './src/cbc/'],
#     ),
#     Extension(
#         "cbc.analog_system",
#         ["src/cbc/analog_system.pyx"],
#     ),
#     Extension(
#         "cbc.digital_control",
#         ["src/cbc/digital_control.pyx"],
#     ),
#     Extension(
#         "cbc.analog_signal",
#         ["src/cbc/analog_signal.pyx"],
#     ),
# ]

compiler_directives = {"language_level": 3, "embedsignature": True}
print(find_packages('src'))

setup(
    name="cbc",  # Replace with your own username
    version="0.0.1",
    author="Hampus Malmberg",
    author_email="hampus.malmberg88@gmail.com",
    description="A toolbox for simulating control-bounded converters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hammal/cbc",
    packages=['cbc'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    zip_safe=False,
)

# setuptools.install_required
