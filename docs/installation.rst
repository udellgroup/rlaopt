Installation
=============

Installation
------------

Please clone this repository. The package can be installed in a Python environment in editable mode using the following command:

.. code-block:: bash

   pip install -e .

Requirements
------------

rlaopt requires:

* Python >= 3.10
* PyTorch >= 2.6.0
* pydantic >= 2.12.0
* tensordict >= 0.10.0
* typing_extensions >= 4.15.0

Optional Dependencies
---------------------

For development, additional dependencies are available:

.. code-block:: bash

   pip install -e ".[dev]"

For testing:

.. code-block:: bash

   pip install -e ".[test]"

Building Documentation
-----------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   make html

The documentation will be generated in ``docs/_build/html/``.
