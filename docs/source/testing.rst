.. _testing:

=======
Testing
=======

To run the tests from a single version of Python, simply run the following command from the root directory of the project (making sure that the Python environment with the ``Dev``-installed version of **rust_nurbs** is active):

.. code-block:: shell

    pytest tests

To ensure that the tests work for all supported Python versions, use this command instead:

.. code-block:: shell

    tox run

.. important::

    The ``Dev`` installation method must be used to acquire all the dependencies required to run these tests.
