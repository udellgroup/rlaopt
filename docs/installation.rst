Installation
=============

Installation
------------

Install ``uv``, clone this repository, and create the development environment using the committed lockfile:

.. code-block:: bash

   uv sync

Requirements
------------

rlaopt requires:

* Python >= 3.10
* PyTorch >= 2.6.0
* pydantic >= 2.12.0
* tensordict >= 0.10.0
* typing_extensions >= 4.15.0

Development Dependencies
------------------------

The default development environment includes the test and pre-commit dependency groups:

.. code-block:: bash

   uv sync

Run the test suite with:

.. code-block:: bash

   uv run pytest

Building Documentation
-----------------------

To build the documentation locally:

.. code-block:: bash

   uv sync --group docs
   uv run make -C docs html

The documentation will be generated in ``docs/_build/html/``.
