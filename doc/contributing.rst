Contributing
============

We welcome contributions from the community! Whether you're fixing a bug,
adding a feature, improving documentation, or suggesting ideas, your help
makes this package better for everyone.

How to Contribute
-----------------

1. **Report Issues**

   * Use the `GitHub issue tracker <https://github.com/openkim/kim-convergence/issues>`_
   * Include a minimal reproducible example for bugs
   * Check if the issue already exists before creating a new one

2. **Suggest Features**

   * Open an issue with the "enhancement" label
   * Describe the use case and expected behavior
   * Consider if it aligns with the package's scope

3. **Submit Code Changes**

   * Fork the repository
   * Create a feature branch from ``main``
   * Make your changes with clear commit messages
   * Add tests for new functionality
   * Ensure all existing tests pass
   * Submit a pull request

4. **Improve Documentation**

   * Fix typos or unclear explanations
   * Add examples for common use cases
   * Improve API documentation
   * Submit as a PR or open an issue with suggestions

Development Setup
-----------------

.. code-block:: bash

   # Clone and install in development mode
   git clone https://github.com/openkim/kim-convergence.git
   cd kim-convergence
   pip install -e ".[doc]"

   # Run tests
   python -m tests

   # Build documentation locally
   cd doc
   make html

Coding Guidelines
-----------------

* Follow PEP 8 style guidelines
* Use type hints where appropriate
* Write docstrings for all public functions
* Include unit tests for new features
* Update documentation when changing behavior

Testing
-------

* All new code should include tests
* Run the full test suite before submitting: ``python -m tests``
* Aim for high test coverage, especially for core functionality

Pull Request Process
--------------------

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add a clear description of changes in the PR
4. Reference any related issues
5. Wait for review and address feedback

Community
---------

Join our discussion on Gitter:

.. raw:: html

   <a href="https://gitter.im/openkim/kim-convergence">
     <img src="https://badges.gitter.im/openkim/kim-convergence.svg"
          alt="Gitter">
   </a>

For questions, suggestions, or collaboration ideas, please get in touch!

Copyright and License
---------------------

.. image:: https://img.shields.io/badge/license-LGPL--2.1--or--later-blue
   :target: LICENSE
   :alt: License: LGPL-2.1-or-later

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

|copyright|

**Contributions**

All contributions are licensed under the same LGPL-2.1+ terms. By submitting
code, you agree to license your work under these terms and confirm you have
the right to do so.
