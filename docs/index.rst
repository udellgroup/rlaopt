===========
SketchyOpts
===========

SketchyOpts is a `JAX <https://jax.readthedocs.io/>`_ library for sketching-based 
methods. 

The primary goal is to provide accessible and efficient implementation of these methods 
to the scientific community. To this end, the library follows design philosophies of 
`JAXopt <https://jaxopt.github.io/>`_ and provides a familiar interface. We hope the 
library facilitates research productivity and leads to new ideas and exciting solutions.  

----

Installation
------------

The latest release of SketchyOpts can be installed using::

   pip install sketchyopts

Quick Example
-------------

Citation
--------

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/support_vector_machine
   tutorials/ridge_regression
   tutorials/l2_logistic_regression

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 2

   api/sketchyopts.solver
   api/sketchyopts.linear_solve
   api/sketchyopts.low_rank_approx
   api/sketchyopts.sketching
   api/sketchyopts.base
   api/sketchyopts.operator
   api/sketchyopts.prox
   api/sketchyopts.util
   api/sketchyopts.error

----

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: 
         :material-regular:`library_books;2em` Tutorials
         :text-align: center
         :link: tutorials/index
         :link-type: doc

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: 
         :material-regular:`science;2em` Benchmarks
         :text-align: center

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: 
         :material-regular:`menu_book;2em` API Reference
         :text-align: center