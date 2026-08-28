Installation
=============

From PyPI
---------

Install the latest release with pip:

.. code-block:: bash

   pip install rlaopt

Or add rlaopt to a uv-managed project:

.. code-block:: bash

   uv add rlaopt

Requirements
------------

rlaopt requires Python 3.10 or newer. Package installers resolve the remaining
runtime dependencies declared in ``pyproject.toml``.

Development Installation
------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, clone
the repository, and create the development environment from the committed
lockfile:

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
