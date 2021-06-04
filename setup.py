"""Setuptools based setup module."""

from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='convergence',
    version=versioneer.get_version(),
    description='convergence - convergence module.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/openkim/convergence',
    author='Yaser Afshar',
    author_email='yafshar@openkim.org',
    license='CDDL-1.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    install_requires=['numpy>=1.16', 'scipy', 'kim_edn', 'joblib'],
    python_requires='>=3.6',
    include_package_data=True,
    keywords='convergence',
    packages=find_packages(),
    cmdclass=versioneer.get_cmdclass(),
)
