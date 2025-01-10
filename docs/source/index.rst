rust_nurbs
==========

Welcome to **rust_nurbs**, a Python API for evaluation of Non-Uniform Rational B-Splines (NURBS) curves and surfaces implemented in Rust. The primary goals of this package are to allow for extremely fast NURBS evaluation while providing a user-friendly interface through Python with zero external dependencies.

Speed Comparison
----------------

Performance can be of high importance when evaluating NURBS objects, especially when evaluating many points on surfaces with high degrees. See the below image for a comparison of the performance of this library versus an essentially identical pure-Python implementation.

.. figure:: images/python-rust-speed-comparison.*
   :width: 500px
   :align: center

   Comparison of the Rust NURBS implementation vs. a pure-Python implementation for several cases

.. list-table:: Case Description
    :widths: 20 60 20
    :header-rows: 1

    * - Case
      - Description
      - Samples/Size
    * - Case 1
      - Degree 10 Bernstein polynomial evaluation at random :math:`t`-values
      - 500,000
    * - Case 2
      - Degree 4 Bézier curve evaluation at random :math:`t`-values
      - 500,000
    * - Case 3
      - Cubic B-spline curve evaluation at random :math:`t`-values
      - 500,000
    * - Case 4
      - :math:`1 \times 4` Bézier surface evaluation at random :math:`(u,v)`-pairs
      - 500,000
    * - Case 5
      - :math:`1 \times 4` Bézier surface grid evaluation
      - :math:`500 \times 500`
    * - Case 6
      - :math:`2 \times 4` NURBS surface of revolution grid evaluation
      - :math:`500 \times 500`
    * - Case 7
      - :math:`2 \times 9` NURBS surface of revolution grid evaluation
      - :math:`500 \times 500`
    

Basic Installation
------------------

Installation is straightforward using the simple

.. code-block:: shell

    pip install rust-nurbs

.. note::

    The command ``pip install rust_nurbs`` also works. This underscore must be used when importing the library inside of Python because a hyphen in an import statement is not valid Python syntax.

If using the ``Stable`` version, the single ``pip install`` command should work for all major operating versions, CPU architectures, and versions of Python ``>=3.8``. See :ref:`install` for more detailed installation instructions if you have an unsupported architecture or are a developer and would like to extend or develop the library.

Quick Start
-----------

After installing **rust_nurbs** in the current environment and starting a Python console (or from inside a ``.py`` script), an example Bézier surface with degree :math:`n=1` and :math:`m=3` can be evaluated at :math:`(u,v) = (0.3, 0.8)` using the following code:

.. code-block:: python

    import rust_nurbs
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]], 
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_point = np.array(rust_nurbs.bezier_surf_eval(p, 0.3, 0.8))

If desired, all the functions listed in the :ref:`api` can be dumped to the current namespace to shorten the function names:

.. code-block:: python

    from rust_nurbs import *
    p = np.array([
        [[0.0, 0.0, 0.0], [0.3, 0.2, 0.0], [0.6, -0.1, 0.0], [1.2, 0.1, 0.0]], 
        [[0.0, 0.0, 1.0], [0.3, 0.4, 1.0], [0.6, -0.2, 1.0], [1.2, 0.2, 1.0]]
    ])
    surf_point = np.array(bezier_surf_eval(p, 0.3, 0.8))

See the `test file <https://github.com/mlau154/rust_nurbs/blob/main/tests/test_rust_nurbs.py>`_ for more examples.

Source & API
------------

The source code can be viewed on GitHub `here <https://github.com/mlau154/rust_nurbs/blob/main/src/lib.rs>`_. See the :ref:`api` section for a detailed Python API reference.

Examples
--------

See the `test file <https://github.com/mlau154/rust_nurbs/blob/main/tests/test_rust_nurbs.py>`_ for example usages of each function.

Contents
--------

.. toctree::

    install
    testing
    api
