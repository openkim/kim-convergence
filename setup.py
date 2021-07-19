"""Setuptools based setup module."""

from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kim-convergence',
    version=versioneer.get_version(),
    description='kim-convergence designed to help in automatic equilibration detection & run length control.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/openkim/kim-convergence',
    author='Yaser Afshar',
    author_email='yafshar@openkim.org',
    license='LGPLv2+',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    install_requires=['numpy>=1.16', 'scipy', 'kim_edn', 'joblib'],
    python_requires='>=3.7',
    include_package_data=True,
    keywords=['convergence', 'automated equilibration detection',
              'run length control', 'upper confidence limit',
              'confidence interval'],
    packages=find_packages(),
    cmdclass=versioneer.get_cmdclass(),
)
