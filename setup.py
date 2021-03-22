from setuptools import Extension, setup
from Cython.Build import cythonize
from glob import glob
import os
import numpy

import os.path


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

USING_CYTHON = True
ext = "pyx"

root_path = "src/cbadc"

parallel_digital_estimator_path = "src/cbadc/parallel_digital_estimator"


source_files = glob(root_path + "/*.pyx")
# for file in glob(root_path + "/*.pyx"):
#     source_files.append(os.path.sep.join(file.split(os.path.sep)[1:]))
# print(source_files)

extensions = [
    Extension(os.path.sep.join(source.split(os.path.sep)[1:]).split('.')[0].
              replace(os.path.sep, '.'),
              sources=[source],
              ) for source in source_files]

# extensions += [
#     Extension(
#         source.split('.')[0].replace(os.path.sep, '.'),
#         sources=[source],
#         # extra_compile_args=["-fopenmp"],
#         language="c++",
#     ) for source in glob(parallel_digital_estimator_path + "/*.pyx")
# ]

compiler_directives = {"language_level": 3, "embedsignature": True}


if USING_CYTHON:
    ext_modules = cythonize(extensions,
                            compiler_directives=compiler_directives,
                            include_path=[
                                numpy.get_include()])
else:
    ext_modules = no_cythonize(extensions)


setup(
    name="cbadc",
    version="0.0.1",
    author="Hampus Malmberg",
    author_email="hampus.malmberg88@gmail.com",
    description="A toolbox for simulating control-bounded converters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hammal/cbadc",
    packages=['cbadc'],
    license="GPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=REQUIREMENTS,
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False,
)

# setuptools.install_required
