rust_nurbs
==========

Welcome to **rust_nurbs**, a Python API for evaluation of Non-Uniform Rational B-Splines (NURBS) curves and surfaces implemented in Rust. The primary goals of this package are to allow for extremely fast NURBS evaluation while providing a user-friendly interface through Python and limiting external dependencies.

Installation
------------

Installation is straightforward using the simple

.. tab-set::

    .. tab-item:: Stable

        .. code-block:: shell

            pip install rust_nurbs
    
    .. tab-item:: Latest

        .. code-block:: shell

            git clone https://github.com/mlau154/rust_nurbs.git
            cd rust_nurbs
            pip install .

However, Rust must be installed properly on your machine for the ``pip install`` command to pass. See :ref:`install`
for more details.

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

Source & API
------------

The source code can be viewed on GitHub `here <https://github.com/mlau154/rust_nurbs/blob/main/src/lib.rs>`_. See the :ref:`api` section for a detailed Python API reference.

Examples
--------

See the `test file <https://github.com/mlau154/rust_nurbs/blob/main/tests/test_rust_nurbs.py>`_ for example usages of each funciton.

Developer
---------

To install the development version of **rust_nurbs** which installs the dependencies required to run the tests, use

.. code-block:: shell

    git clone https://github.com/mlau154/rust_nurbs.git
    cd rust_nurbs
    pip install .[dev]

To run the tests from a single version of Python, simply run the following command from the root directory of the project (making sure to first activate the Python environment):

.. code-block:: shell

    pytest tests

To ensure that the tests work for all supported Python versions, use this command instead:

.. code-block:: shell

    tox run

Contents
--------

.. toctree::

    install
    api
